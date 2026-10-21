"""
Flask Application for HeteroSched Performance Dashboard.
Provides secure, local REST APIs to inspect system specifications, query approved traces,
execute real C++/CUDA benchmarks, stream results, and export clean/raw CSVs.
Also serves the built frontend production bundle if present.
"""

import os
from datetime import datetime, timezone
from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS

import importlib.util
import sys

# Ensure current server directory is on sys.path for direct module resolution
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))

if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

if os.path.isdir(os.path.join(_PARENT_DIR, "server")) and _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# Deterministically select package-qualified vs flat module imports based on
# whether 'server' resolves to this directory in sys.path.
# This eliminates broad try/except ImportError blocks and ensures internal syntax,
# runtime, or missing-dependency errors inside parser/runner are never masked.
_use_server_package = False
_server_spec = importlib.util.find_spec("server")
if _server_spec is not None and _server_spec.submodule_search_locations:
    for _loc in _server_spec.submodule_search_locations:
        if os.path.abspath(_loc) == _CURRENT_DIR:
            _use_server_package = True
            break

if _use_server_package:
    from server.parser import parse_benchmark_csv, generate_clean_csv
    from server.runner import (
        APPROVED_TRACES,
        BINARY_PATH,
        WORKSPACE_ROOT,
        execute_benchmark,
        get_approved_traces,
        get_current_status,
    )
else:
    from parser import parse_benchmark_csv, generate_clean_csv
    from runner import (
        APPROVED_TRACES,
        BINARY_PATH,
        WORKSPACE_ROOT,
        execute_benchmark,
        get_approved_traces,
        get_current_status,
    )

app = Flask(__name__)

# Restrict CORS to local frontend development origins
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ]
        }
    },
)

# In-memory storage for latest run data
_LATEST_RUN = {
    "is_fresh": False,
    "source": "none",
    "run_id": None,
    "trace_key": None,
    "trace_name": None,
    "timestamp_iso": None,
    "elapsed_seconds": 0.0,
    "output_csv_path": None,
    "summary": None,
    "tasks": [],
    "stdout": "",
    "stderr": "",
}


def _load_initial_results():
    """Load existing workspace results.csv or server/sample_results.csv as baseline data."""
    default_csv = os.path.join(WORKSPACE_ROOT, "results.csv")
    sample_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_results.csv")

    target_csv = None
    is_sample = False
    if os.path.exists(default_csv):
        target_csv = default_csv
    elif os.path.exists(sample_csv):
        target_csv = sample_csv
        is_sample = True

    if target_csv:
        try:
            tasks, summary = parse_benchmark_csv(target_csv)
            mtime = os.path.getmtime(target_csv)
            mtime_iso = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            _LATEST_RUN["is_fresh"] = False
            _LATEST_RUN["source"] = "demo_sample" if is_sample else "saved_results"
            _LATEST_RUN["run_id"] = "sample_baseline" if is_sample else "saved_baseline"
            _LATEST_RUN["trace_key"] = "mixed"
            _LATEST_RUN["trace_name"] = "Mixed Workload (Demo / Sample Data - Not Live GPU Measurement)" if is_sample else "Mixed Workload (Saved Baseline)"
            _LATEST_RUN["timestamp_iso"] = mtime_iso
            _LATEST_RUN["output_csv_path"] = target_csv
            _LATEST_RUN["summary"] = summary
            _LATEST_RUN["tasks"] = tasks
            _LATEST_RUN["stdout"] = "Demo / Sample benchmark data (offline local GPU measurement)" if is_sample else "Loaded from existing workspace results.csv"
            print(f"Loaded {target_csv} ({len(tasks)} tasks). Marked as {_LATEST_RUN['source']}.")
        except Exception as e:
            print(f"Could not load initial results: {e}")


_load_initial_results()


@app.route("/api/system/info", methods=["GET"])
def get_system_info():
    """Returns detected GPU and system configuration specs, differentiating local GPU from cloud demo."""
    binary_available = os.path.exists(BINARY_PATH)
    mode = "local_gpu" if binary_available else "cloud_demo"
    return jsonify({
        "status": "online",
        "mode": mode,
        "is_local_gpu": binary_available,
        "execution_target": "NVIDIA GeForce RTX 4050 (Direct CUDA Kernel Dispatch)" if binary_available else "Cloud Serverless (Demo / Offline Baseline Only)",
        "gpu": {
            "name": "NVIDIA GeForce RTX 4050 Laptop GPU",
            "compute_capability": "8.9",
            "global_memory_gb": 6.0,
            "sm_count": 20,
            "memory_bus_width_bits": 96,
            "stream_priority_range": "[-5, 0]",
            "hardware_present": binary_available,
        },
        "scheduler": {
            "engine": "HeteroSched Dual-Tier Engine",
            "tier1_fastpath": "Threshold-based filter (<5us)",
            "tier2_predictor": "Compile-embedded Decision Forest / GBDT",
            "cpu_workers": 4,
            "gpu_streams": 4,
            "dynamic_penalties": True,
            "cross_layer_priority": True,
        },
        "host": {
            "platform": "Linux (Ubuntu 24.04 LTS under WSL2)",
            "architecture": "x86_64",
            "mode": mode,
        },
    })


@app.route("/api/traces", methods=["GET"])
def list_traces():
    """Returns the list of whitelisted, approved workload traces."""
    return jsonify({
        "traces": get_approved_traces(),
        "default": "mixed",
    })


@app.route("/api/benchmark/status", methods=["GET"])
def benchmark_status():
    """Checks whether a benchmark run is currently in progress."""
    return jsonify(get_current_status())


@app.route("/api/benchmark/run", methods=["POST"])
def run_benchmark():
    """
    Executes a real benchmark run on the approved workload trace.
    Validates input strictly against the trace whitelist.
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON request body. Expected {'trace': 'trace_key'}."}), 400

    trace_key = data.get("trace")
    if not trace_key or not isinstance(trace_key, str):
        return jsonify({"error": "Missing 'trace' string field in request body."}), 400

    trace_key = trace_key.strip().lower()
    if trace_key not in APPROVED_TRACES:
        allowed = list(APPROVED_TRACES.keys())
        return jsonify({
            "error": f"Invalid trace '{trace_key}'. Approved traces are: {allowed}"
        }), 400

    try:
        run_meta = execute_benchmark(trace_key, timeout_seconds=60.0)
    except FileNotFoundError as e:
        return jsonify({
            "error": "Real benchmark execution requires the local C++/CUDA runtime and NVIDIA GPU hardware (heterosched_bench binary not found). Real GPU benchmarks cannot be run in the cloud serverless environment.",
            "detail": str(e),
            "cloud_environment": True,
        }), 503
    except RuntimeError as e:
        msg = str(e)
        if "progress" in msg.lower():
            return jsonify({"error": msg}), 409
        return jsonify({"error": msg}), 500
    except TimeoutError as e:
        return jsonify({"error": str(e)}), 504
    except Exception as e:
        return jsonify({"error": f"Unexpected execution error: {str(e)}"}), 500

    # Parse output CSV safely
    try:
        tasks, summary = parse_benchmark_csv(run_meta["output_csv_path"])
    except Exception as e:
        return jsonify({
            "error": f"Failed to parse generated benchmark CSV: {str(e)}",
            "stdout": run_meta.get("stdout", ""),
            "stderr": run_meta.get("stderr", ""),
        }), 500

    # Store latest run with explicit fresh flag and source identifier
    _LATEST_RUN["is_fresh"] = True
    _LATEST_RUN["source"] = "fresh_run"
    _LATEST_RUN["run_id"] = run_meta["run_id"]
    _LATEST_RUN["trace_key"] = run_meta["trace_key"]
    _LATEST_RUN["trace_name"] = run_meta["trace_name"]
    _LATEST_RUN["timestamp_iso"] = run_meta["timestamp_iso"]
    _LATEST_RUN["elapsed_seconds"] = run_meta["elapsed_seconds"]
    _LATEST_RUN["output_csv_path"] = run_meta["output_csv_path"]
    _LATEST_RUN["summary"] = summary
    _LATEST_RUN["tasks"] = tasks
    _LATEST_RUN["stdout"] = run_meta["stdout"]
    _LATEST_RUN["stderr"] = run_meta["stderr"]

    return jsonify({
        "success": True,
        "run": _LATEST_RUN,
    })


@app.route("/api/benchmark/latest", methods=["GET"])
def get_latest_benchmark():
    """Returns the most recent benchmark run, distinguishing fresh runs from saved baselines."""
    if _LATEST_RUN["summary"] is None:
        return jsonify({"error": "No benchmark results available yet."}), 404
    return jsonify(_LATEST_RUN)


@app.route("/api/benchmark/download", methods=["GET"])
def download_csv():
    """
    Downloads benchmark results CSV.
    Query parameter:
      - mode='clean' (default): returns standard CSV without '# total_time_ms' comment.
      - mode='raw': returns the raw output file preserving any comment lines.
    """
    mode = request.args.get("mode", "clean").strip().lower()
    csv_path = _LATEST_RUN.get("output_csv_path")

    if not csv_path or not os.path.exists(csv_path):
        _load_initial_results()
        csv_path = _LATEST_RUN.get("output_csv_path")

    if not csv_path or not os.path.exists(csv_path):
        return jsonify({"error": "No benchmark results file available to download."}), 404

    run_id = _LATEST_RUN.get("run_id") or "heterosched_results"

    if mode == "raw":
        with open(csv_path, "r", encoding="utf-8") as f:
            content = f.read()
        filename = f"{run_id}_raw.csv"
    else:
        # Generate clean CSV
        content = generate_clean_csv(_LATEST_RUN.get("tasks", []))
        filename = f"{run_id}_clean.csv"

    return Response(
        content,
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Cache-Control": "no-cache",
        },
    )


# Serve built frontend static bundle if available
FRONTEND_DIST = os.path.join(WORKSPACE_ROOT, "frontend", "dist")


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path.startswith("api/"):
        return jsonify({"error": "API route not found"}), 404
    if os.path.exists(os.path.join(FRONTEND_DIST, path)) and path != "":
        return send_from_directory(FRONTEND_DIST, path)
    if os.path.exists(os.path.join(FRONTEND_DIST, "index.html")):
        return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify({
        "status": "online",
        "service": "HeteroSched API Server",
        "frontend": "Frontend dev server running at http://localhost:5173",
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting HeteroSched Backend API on http://127.0.0.1:{port} ...")
    app.run(host="127.0.0.1", port=port, debug=False)
