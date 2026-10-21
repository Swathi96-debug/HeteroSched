#!/usr/bin/env bash
# ==============================================================================
# HeteroSched Local GPU Dashboard Launcher
# Starts the C++/CUDA-backed Flask API and serves the production React dashboard
# on http://127.0.0.1:<PORT> with direct access to the NVIDIA RTX 4050 GPU.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${WORKSPACE_ROOT}"

PORT="${PORT:-5000}"
RESTART=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--restart)
            RESTART=true
            shift
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-r|--restart] [-p|--port <port>]"
            echo ""
            echo "Options:"
            echo "  -r, --restart   Restart an existing HeteroSched dashboard server"
            echo "  -p, --port      Specify a custom port (default: 5000 or \$PORT)"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage."
            exit 1
            ;;
    esac
done

export PORT

echo "=================================================================="
echo "    HeteroSched: Adaptive Two-Tier Heterogeneous Task Scheduler   "
echo "                Local GPU Telemetry & Benchmark Dashboard         "
echo "=================================================================="

# Check if port is already occupied
OCCUPIED_PID=""
if command -v lsof >/dev/null 2>&1; then
    OCCUPIED_PID=$(lsof -ti :"${PORT}" 2>/dev/null | head -n 1 || true)
elif command -v fuser >/dev/null 2>&1; then
    OCCUPIED_PID=$(fuser "${PORT}"/tcp 2>/dev/null | awk '{print $1}' || true)
fi

if [ -n "${OCCUPIED_PID}" ]; then
    PROC_CMD=$(ps -p "${OCCUPIED_PID}" -o cmd= 2>/dev/null || true)
    if echo "${PROC_CMD}" | grep -q "server/app.py"; then
        if [ "${RESTART}" = true ]; then
            echo "[!] Found existing HeteroSched instance on port ${PORT} (PID ${OCCUPIED_PID})."
            echo "    Gracefully stopping previous instance before restart..."
            kill -15 "${OCCUPIED_PID}" 2>/dev/null || true
            for i in {1..15}; do
                if ! kill -0 "${OCCUPIED_PID}" 2>/dev/null; then
                    break
                fi
                sleep 0.2
            done
            if kill -0 "${OCCUPIED_PID}" 2>/dev/null; then
                kill -9 "${OCCUPIED_PID}" 2>/dev/null || true
            fi
            sleep 0.5
            echo "[✓] Previous HeteroSched instance stopped successfully."
        else
            echo "=================================================================="
            echo "[✓] HeteroSched dashboard server is ALREADY ACTIVE on http://127.0.0.1:${PORT}"
            echo "    Running Process: PID ${OCCUPIED_PID} (${PROC_CMD})"
            echo "=================================================================="
            echo "Opening http://127.0.0.1:${PORT} in your browser..."
            if command -v powershell.exe >/dev/null 2>&1; then
                powershell.exe -NoProfile -Command "Start-Process 'http://127.0.0.1:${PORT}'" >/dev/null 2>&1 &
            fi
            echo ""
            echo "To restart the server with fresh code, run:"
            echo "    ./scripts/launch_dashboard.sh --restart"
            echo "Or run on a different port:"
            echo "    PORT=5001 ./scripts/launch_dashboard.sh"
            echo "=================================================================="
            exit 0
        fi
    else
        echo "=================================================================="
        echo "[!] ERROR: Port ${PORT} is currently occupied by an unrelated program!"
        echo "    PID: ${OCCUPIED_PID}"
        echo "    Command: ${PROC_CMD}"
        echo ""
        echo "HeteroSched will not automatically terminate unrelated processes."
        echo ""
        echo "Safe Recovery Options:"
        echo "  1. Run HeteroSched on an alternative port:"
        echo "       ./scripts/launch_dashboard.sh --port 5001"
        echo "     or:"
        echo "       PORT=5001 ./scripts/launch_dashboard.sh"
        echo ""
        echo "  2. Or stop the conflicting process if it is no longer needed:"
        echo "       kill -15 ${OCCUPIED_PID}"
        echo "=================================================================="
        exit 1
    fi
fi

# 1. Verify NVIDIA GPU and CUDA Driver
echo -n "[1/4] Checking NVIDIA GPU hardware... "
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true)
    if [ -n "${GPU_INFO}" ]; then
        echo "FOUND: ${GPU_INFO}"
    else
        echo "WARNING: nvidia-smi found but could not query device."
    fi
else
    echo "WARNING: nvidia-smi not detected in PATH."
fi

# 2. Verify or build the C++/CUDA benchmark executable
echo -n "[2/4] Verifying C++/CUDA binary (build/heterosched_bench)... "
if [ ! -f "build/heterosched_bench" ]; then
    echo "Not found. Building..."
    mkdir -p build
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    make -C build heterosched_bench -j4
    echo "Binary successfully compiled."
else
    echo "READY."
fi

# 3. Verify Python runtime and dependencies
echo -n "[3/4] Checking Python environment... "
PYTHON_BIN="python3"
if [ -f ".venv/bin/python3" ]; then
    PYTHON_BIN=".venv/bin/python3"
fi

if ! ${PYTHON_BIN} -c "import flask, flask_cors" >/dev/null 2>&1; then
    echo "Installing missing dependencies from server/requirements.txt..."
    ${PYTHON_BIN} -m pip install -r server/requirements.txt
fi
echo "READY ($( ${PYTHON_BIN} --version ))."

# 4. Verify Frontend Production Assets
echo -n "[4/4] Verifying Frontend dashboard build... "
if [ ! -f "frontend/dist/index.html" ]; then
    echo "Building production frontend assets..."
    (cd frontend && npm run build)
else
    echo "READY."
fi

echo "=================================================================="
echo " Starting HeteroSched unified server on: http://127.0.0.1:${PORT}"
echo " Mode: LOCAL GPU (Direct dispatch to NVIDIA GeForce RTX 4050)"
echo " Press Ctrl+C to stop the dashboard server."
echo "=================================================================="

# Proactively open default browser on Windows host if running under WSL
if command -v powershell.exe >/dev/null 2>&1; then
    (sleep 1.5 && powershell.exe -NoProfile -Command "Start-Process 'http://127.0.0.1:${PORT}'") >/dev/null 2>&1 &
fi

exec ${PYTHON_BIN} server/app.py
