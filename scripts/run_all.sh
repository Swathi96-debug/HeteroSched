#!/usr/bin/env bash
set -euo pipefail

echo "=============================================="
echo " Starting HeteroSched Automated Workflow"
echo "=============================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# 1. Setup Environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif command -v python3 &>/dev/null; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
fi

# 2. Train Models and Export to C++
echo "[1/4] Generating ML dataset, training models, and exporting..."
python3 ml/dataset_generator.py
python3 ml/train_model.py
python3 ml/export_to_cpp.py

# 3. Generate Benchmark Traces
echo "[2/4] Generating benchmark trace definitions..."
mkdir -p benchmarks/traces
python3 benchmarks/trace_generator.py

# 4. Build Engine
echo "[3/4] Compiling native HeteroSched runtime engine..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"
cd ..

# 5. Run Experiments and Generate Plots
echo "[4/4] Executing comparative benchmarks..."
./build/heterosched_bench --trace benchmarks/traces/mixed_trace.json --output results.csv
mkdir -p artifacts
python3 scripts/plot_results.py --input results.csv --output-dir artifacts/

echo "=============================================="
echo " HeteroSched Execution Complete. Results in artifacts/"
echo "=============================================="
