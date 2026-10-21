"""
Runner module for HeteroSched benchmarks.
Enforces thread-safe single-execution semantics, strict workload whitelisting,
timeout protection, and error capture.
"""

import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BINARY_PATH = os.path.join(WORKSPACE_ROOT, "build", "heterosched_bench")
RUNS_DIR = os.path.join(WORKSPACE_ROOT, "server", "runs")

# Ensure runs directory exists (fallback to /tmp in read-only serverless environments)
try:
    os.makedirs(RUNS_DIR, exist_ok=True)
except OSError:
    RUNS_DIR = os.path.join('/tmp', 'heterosched_runs')
    try:
        os.makedirs(RUNS_DIR, exist_ok=True)
    except OSError:
        pass

# Strict trace whitelist: key -> metadata and safe relative path
APPROVED_TRACES: Dict[str, Dict[str, Any]] = {
    "mixed": {
        "key": "mixed",
        "name": "Mixed Workload",
        "file": os.path.join(WORKSPACE_ROOT, "benchmarks", "traces", "mixed_trace.json"),
        "description": "200 heterogeneous tasks spanning SGEMM, SPMV, CONV2D, and QuickSort with varied compute-to-transfer ratios.",
        "task_count": 200,
        "tags": ["Heterogeneous", "Balanced", "Multi-Kernel"],
    },
    "bursty": {
        "key": "bursty",
        "name": "Bursty Arrival Workload",
        "file": os.path.join(WORKSPACE_ROOT, "benchmarks", "traces", "bursty_trace.json"),
        "description": "150 tasks with dense burst arrival patterns to evaluate host dispatcher responsiveness and queue congestion penalties.",
        "task_count": 150,
        "tags": ["Queue Congestion", "Bursty", "High Priority"],
    },
    "pcie_heavy": {
        "key": "pcie_heavy",
        "name": "PCIe-Heavy Saturation Workload",
        "file": os.path.join(WORKSPACE_ROOT, "benchmarks", "traces", "pcie_heavy_trace.json"),
        "description": "100 memory-intensive tasks with large buffer payloads to challenge PCIe bus saturation thresholds and dynamic offload penalties.",
        "task_count": 100,
        "tags": ["Memory Bound", "PCIe Saturation", "Data Intensive"],
    },
}

# Process execution lock to prevent concurrent runs
_BENCHMARK_LOCK = threading.Lock()
# Dedicated status lock to allow prompt, thread-safe status polling while benchmark executes
_STATUS_LOCK = threading.Lock()
_CURRENT_STATUS = {
    "is_running": False,
    "current_trace": None,
    "start_time": None,
}


def get_current_status() -> Dict[str, Any]:
    with _STATUS_LOCK:
        return dict(_CURRENT_STATUS)


def get_approved_traces() -> Dict[str, Dict[str, Any]]:
    return {
        k: {
            "key": v["key"],
            "name": v["name"],
            "description": v["description"],
            "task_count": v["task_count"],
            "tags": v["tags"],
        }
        for k, v in APPROVED_TRACES.items()
    }


def execute_benchmark(trace_key: str, timeout_seconds: float = 60.0) -> Dict[str, Any]:
    """
    Executes the real C++/CUDA heterosched_bench binary against the requested approved trace.
    Guarantees:
      - Only approved trace keys are accepted.
      - Binary path is hardcoded; no shell invocation.
      - Single-flight execution enforced by mutex lock.
      - Output is saved to an immutable, timestamped run file in server/runs/.
    """
    if trace_key not in APPROVED_TRACES:
        allowed = list(APPROVED_TRACES.keys())
        raise ValueError(f"Invalid trace '{trace_key}'. Approved traces are: {allowed}")

    if not os.path.exists(BINARY_PATH):
        raise FileNotFoundError(
            f"HeteroSched binary not found at '{BINARY_PATH}'. Please ensure build/heterosched_bench exists."
        )

    trace_info = APPROVED_TRACES[trace_key]
    trace_file = trace_info["file"]

    if not os.path.exists(trace_file):
        raise FileNotFoundError(f"Trace file not found: {trace_file}")

    # Acquire lock without blocking indefinitely
    acquired = _BENCHMARK_LOCK.acquire(blocking=False)
    if not acquired:
        raise RuntimeError("Another benchmark run is currently in progress. Please wait.")

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp_str}_{trace_key}"
    output_csv_path = os.path.join(RUNS_DIR, f"{run_id}.csv")

    with _STATUS_LOCK:
        _CURRENT_STATUS["is_running"] = True
        _CURRENT_STATUS["current_trace"] = trace_key
        _CURRENT_STATUS["start_time"] = time.time()

    cmd = [
        BINARY_PATH,
        "--trace",
        trace_file,
        "--output",
        output_csv_path,
    ]

    start_mono = time.monotonic()
    stdout_text = ""
    stderr_text = ""
    returncode = -1

    try:
        proc = subprocess.run(
            cmd,
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        returncode = proc.returncode
        stdout_text = proc.stdout
        stderr_text = proc.stderr
    except subprocess.TimeoutExpired as e:
        stdout_text = e.stdout or "" if hasattr(e, "stdout") else ""
        stderr_text = f"Benchmark timed out after {timeout_seconds} seconds."
        raise TimeoutError(f"Benchmark execution exceeded {timeout_seconds}s limit.")
    except Exception as e:
        stderr_text = f"Execution exception: {str(e)}"
        raise RuntimeError(f"Failed to execute benchmark: {str(e)}")
    finally:
        elapsed_sec = time.monotonic() - start_mono
        with _STATUS_LOCK:
            _CURRENT_STATUS["is_running"] = False
            _CURRENT_STATUS["current_trace"] = None
            _CURRENT_STATUS["start_time"] = None
        _BENCHMARK_LOCK.release()

    if returncode != 0:
        error_msg = stderr_text.strip() or stdout_text.strip() or f"Process exited with code {returncode}"
        raise RuntimeError(f"Benchmark binary failed (code {returncode}): {error_msg}")

    if not os.path.exists(output_csv_path):
        raise FileNotFoundError(f"Expected benchmark output CSV was not created: {output_csv_path}")

    return {
        "run_id": run_id,
        "trace_key": trace_key,
        "trace_name": trace_info["name"],
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_sec, 3),
        "output_csv_path": output_csv_path,
        "stdout": stdout_text,
        "stderr": stderr_text,
    }
