# HeteroSched: Adaptive Two-Tier Heterogeneous Task Scheduler

HeteroSched is an adaptive two-tier heterogeneous CPU-GPU task scheduling engine designed for systems with mixed compute and PCIe transfer characteristics. It combines a low-latency threshold-based fast-path filter (<5 µs) with a compile-embedded decision forest model, dynamically routing tasks across CPU worker threads and CUDA streams to maximize overall throughput and minimize synchronization latency.

Verified and tested on **NVIDIA GeForce RTX 4050 Laptop GPU** (Compute Capability 8.9, CUDA 13.0) under **Ubuntu 24.04 LTS (WSL2)**.

---

## Architecture Highlights

- **Tier 1 (Micro-Heuristic Fast-Path)**: Evaluates task parameters (`dim_m, dim_n, dim_k`, transfer volume) in <5 µs, directing non-GPU workloads (e.g. quicksort, string tokenization) directly to CPU worker threads.
- **Tier 2 (Embedded Decision Forest)**: Classifies compute-intensive kernels with learned placement policies embedded directly into C++ header code.
- **Hardware Telemetry & Runtime**: Direct CUDA stream concurrency (`stream_id` 0–3) and multi-core CPU thread worker scheduling with nanosecond-level clock measurement.
- **Live React Telemetry Dashboard**: Real-time Gantt/timeline inspection, throughput charts, device splits, and one-click benchmark execution.

---

## System Requirements

- **Operating System**: Linux (Ubuntu 22.04 / 24.04 LTS or WSL2 Ubuntu)
- **GPU Hardware**: Compatible NVIDIA GPU (tested on NVIDIA GeForce RTX 4050 Laptop GPU, 6GB VRAM)
- **CUDA Toolchain**: NVIDIA CUDA Toolkit >= 12.0 with `nvcc` and compatible NVIDIA display drivers (tested on Driver 581.86, CUDA 13.0)
- **Compiler**: C++17 compliant compiler (`g++` >= 11 or `clang++` >= 11)
- **Build Tools**: CMake >= 3.18, Make
- **Python**: Python >= 3.10 with `pip`
- **Node.js**: Node.js >= 18 and `npm`

---

## Quickstart: Unified Local Dashboard

The simplest way to start the entire system (backend API and compiled frontend dashboard) is using the unified launch script:

### From Linux / WSL2:
```bash
# 1. Clone repository:
git clone https://github.com/Swathi96-debug/HeteroSched.git
cd HeteroSched

# 2. Launch dashboard:
./scripts/launch_dashboard.sh
```

### From Windows:
Double-click `launch_dashboard.bat` from the repository root, or run:
```cmd
launch_dashboard.bat
```

The script will automatically:
1. Verify NVIDIA GPU hardware and CUDA driver accessibility.
2. Compile the native `build/heterosched_bench` C++/CUDA binary if missing.
3. Install required Python server dependencies (`server/requirements.txt`).
4. Build the production React frontend bundle if not yet built.
5. Check for port conflicts gracefully:
   - If an existing HeteroSched instance is already running on port 5000, it opens the browser directly and prints management options.
   - If an unrelated process occupies port 5000, it reports the conflicting PID and refuses to terminate it.
6. Launch the dashboard at **http://127.0.0.1:5000** and open your default browser.

### Launcher Command Options:
```bash
./scripts/launch_dashboard.sh --help          # Show options
./scripts/launch_dashboard.sh --restart       # Restart an existing HeteroSched instance
./scripts/launch_dashboard.sh --port 5001     # Launch on a custom port
PORT=5001 ./scripts/launch_dashboard.sh       # Alternative environment variable syntax
```

---

## Manual Setup and Build Instructions

If you prefer building and running components individually:

### 1. Build the C++/CUDA Benchmark Binary
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make heterosched_bench -j$(nproc)
cd ..
```
The compiled binary will be located at `build/heterosched_bench`.

### 2. Build the Production Frontend
```bash
cd frontend
npm install
npm run build
cd ..
```

### 3. Start the Flask Backend Server
```bash
# Create/activate virtual environment:
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies:
pip install -r server/requirements.txt

# Run server:
python3 server/app.py
```
Open **http://127.0.0.1:5000** to view the dashboard.

---

## Workload Traces & Executing Benchmarks

HeteroSched provides three approved benchmark workload traces in `benchmarks/traces/`:

1. **Mixed Workload (`mixed_trace.json`)**: 200 tasks balancing compute-intensive kernels (SGEMM, CONV2D, SPMV, FFT) with CPU-bound tasks (QUICKSORT, STRING_TOKEN). Demonstrates dual-tier placement accuracy (e.g. 66 CPU / 134 GPU).
2. **PCIe-Heavy Workload (`pcie_heavy_trace.json`)**: 100 large-matrix tasks (up to 4096×4096) stressing host-to-device and device-to-host PCIe bus transfers.
3. **Bursty Workload (`bursty_trace.json`)**: Sudden arrival bursts testing queueing delays and fast-path decision latency under saturation.

### Executing via Dashboard
1. Open the dashboard at `http://127.0.0.1:5000`.
2. Verify the status indicator displays **🟢 Local GPU Engine Active (RTX 4050)**.
3. Select an approved workload from the dropdown.
4. Click **"Execute Real Benchmark (RTX 4050)"**.
5. The dashboard executes the compiled binary live, captures task placement, timings, and GPU stream allocations, and updates with a **FRESH REAL-TIME RUN** badge.

### Executing via CLI
You can also execute the C++/CUDA benchmark directly from the terminal:
```bash
./build/heterosched_bench --trace benchmarks/traces/mixed_trace.json --output results.csv
```

---

## CSV Export Formats

Benchmark results can be exported from the dashboard or downloaded via API:

- **Clean CSV** (`/api/benchmark/download?mode=clean`):
  Standard tabular CSV containing only per-task data rows. Ready for direct ingestion into pandas, R, Excel, or analytical pipelines.
- **Raw CSV** (`/api/benchmark/download?mode=raw`):
  Includes system metadata header comments (`# total_time_ms`, `# config`, hardware parameters).

### Schema:
| Column | Description |
|---|---|
| `task_id` | Sequential task identifier |
| `type` | Kernel type (`SGEMM`, `CONV2D`, `SPMV`, `FFT`, `QUICKSORT`, `STRING_TOKEN`) |
| `priority` | Task priority tier (0 = normal, 1 = high, 2 = critical) |
| `dim_m`, `dim_n`, `dim_k` | Matrix / tensor problem dimensions |
| `device` | Scheduled execution device (`CPU` or `GPU`) |
| `t_decision_us` | Latency taken by HeteroSched scheduler decision (µs) |
| `t_transfer_us` | Host-to-device and device-to-host memory transfer duration (µs) |
| `t_exec_us` | Kernel execution duration (µs) |
| `t_sync_us` | Synchronization barrier wait time (µs) |
| `t_total_latency_us` | End-to-end task completion latency (µs) |
| `stream_id` | Assigned CUDA stream (0–3 for GPU, -1 for CPU) |
| `cpu_core` | Assigned CPU thread worker (0–3 for CPU, -1 for GPU) |

---

## Local GPU Mode vs. Cloud Demo Mode

- **Local GPU Mode**: When running locally with access to an NVIDIA GPU, the backend executes the real compiled binary (`build/heterosched_bench`) with live CUDA kernel dispatch.
- **Cloud Demo Mode (Vercel)**: Hosted instances (e.g. `https://heterosched-cqzm.vercel.app/`) run in serverless cloud environments without GPU hardware. In cloud mode, the dashboard clearly displays **☁️ Cloud Demo Mode (Vercel Serverless)** and uses offline baseline results for inspection. Real GPU execution cannot be performed in cloud serverless sandboxes.

---

## Running Automated Tests

### Python Server and Integration Tests:
```bash
python3 -m unittest discover tests
```

### C++ Unit and Regression Tests:
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make test_phase_a test_phase_b test_phase_c -j$(nproc)

./test_phase_a
./test_phase_c
./test_phase_b
```
