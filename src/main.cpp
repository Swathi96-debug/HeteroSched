#include "heterosched_api.h"
#include "workload_types.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <cstdlib>

bool allocate_task_buffers(TaskDescriptor* task);
void free_task_buffers(TaskDescriptor* task);
void gpu_warmup();
void print_gpu_info();

struct TraceEntry {
    uint64_t id;
    std::string type;
    uint32_t priority;
    size_t dim_m, dim_n, dim_k;
    size_t bytes_in, bytes_out;
    uint64_t estimated_flops;
};

static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n\"");
    size_t end = s.find_last_not_of(" \t\r\n\"");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

static std::vector<TraceEntry> load_trace(const std::string& path) {
    std::vector<TraceEntry> entries;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open trace file: " << path << "\n";
        return entries;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    TraceEntry current;
    bool in_object = false;

    std::istringstream stream(content);
    std::string line;

    while (std::getline(stream, line)) {
        line = trim(line);
        if (line.empty() || line == "[" || line == "]") continue;

        if (line == "{") {
            in_object = true;
            current = TraceEntry{};
            continue;
        }

        if (line == "}" || line == "},") {
            if (in_object) {
                entries.push_back(current);
                in_object = false;
            }
            continue;
        }

        if (!in_object) continue;

        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;

        std::string key = trim(line.substr(0, colon));
        std::string val = trim(line.substr(colon + 1));
        if (!val.empty() && val.back() == ',') val.pop_back();
        val = trim(val);

        if (key == "id")              current.id = std::stoull(val);
        else if (key == "type")       current.type = val;
        else if (key == "priority")   current.priority = std::stoul(val);
        else if (key == "dim_m")      current.dim_m = std::stoull(val);
        else if (key == "dim_n")      current.dim_n = std::stoull(val);
        else if (key == "dim_k")      current.dim_k = std::stoull(val);
        else if (key == "bytes_in")   current.bytes_in = std::stoull(val);
        else if (key == "bytes_out")  current.bytes_out = std::stoull(val);
        else if (key == "estimated_flops") current.estimated_flops = std::stoull(val);
    }

    return entries;
}

static void write_results_csv(const std::string& path,
                               const std::vector<TaskDescriptor>& tasks,
                               double total_time_ms,
                               const HeteroSchedConfig& config) {
    std::ofstream out(path);
    out << "task_id,type,priority,dim_m,dim_n,dim_k,"
        << "device,t_decision_us,t_transfer_us,t_exec_us,"
        << "t_sync_us,t_total_latency_us,stream_id,cpu_core\n";

    for (auto& t : tasks) {
        out << t.task_id << ","
            << TaskTypeToString(t.type) << ","
            << t.priority << ","
            << t.dim_m << "," << t.dim_n << "," << t.dim_k << ","
            << DeviceToString(t.metrics.assigned_device) << ","
            << t.metrics.t_decision_us << ","
            << t.metrics.t_transfer_us << ","
            << t.metrics.t_exec_us << ","
            << t.metrics.t_sync_us << ","
            << t.metrics.t_total_latency_us << ","
            << t.metrics.stream_id_used << ","
            << t.metrics.cpu_core_used << "\n";
    }

    out << "\n# total_time_ms," << total_time_ms << "\n";
    out << "# config," << GetConfigDescription(config) << "\n";
    out << "# notes,t_total_latency_us measures synchronous end-to-end task completion latency; failed tasks record 0.0\n";
    out.close();
}

static void print_summary(const std::vector<TaskDescriptor>& tasks, double total_time_ms,
                          const HeteroSchedConfig& config, const GatedController* controller = nullptr) {
    size_t cpu_count = 0, gpu_count = 0, failed_count = 0;
    double total_decision = 0, total_exec = 0, total_transfer = 0, total_latency = 0;

    for (auto& t : tasks) {
        if (!t.metrics.success) {
            failed_count++;
            continue;
        }
        if (t.metrics.assigned_device == ExecutionDevice::DEVICE_CPU) cpu_count++;
        else gpu_count++;
        total_decision += t.metrics.t_decision_us;
        total_exec += t.metrics.t_exec_us;
        total_transfer += t.metrics.t_transfer_us;
        total_latency += t.metrics.t_total_latency_us;
    }

    size_t completed = cpu_count + gpu_count;
    std::cout << "\n=== HeteroSched Results ===\n"
              << "Mode: " << GetConfigDescription(config) << "\n"
              << "Total tasks: " << tasks.size() << "\n"
              << "  Completed: " << completed << " (CPU: " << cpu_count << "  GPU: " << gpu_count << ")\n"
              << "  Failed: " << failed_count << "\n"
              << "Total wall time: " << total_time_ms << " ms\n"
              << "Throughput: " << (completed / (total_time_ms / 1000.0)) << " tasks/sec\n"
              << "Avg decision overhead: " << (completed > 0 ? total_decision / completed : 0) << " us\n"
              << "Avg exec time: " << (completed > 0 ? total_exec / completed : 0) << " us\n"
              << "Avg total completion latency: " << (completed > 0 ? total_latency / completed : 0) << " us\n";
    if (gpu_count > 0) {
        std::cout << "Avg transfer time (GPU tasks): " << (total_transfer / gpu_count) << " us\n";
    } else {
        std::cout << "Avg transfer time (GPU tasks): N/A (no GPU tasks executed)\n";
    }

    if (controller && config.enable_tier2 && !config.force_cpu && !config.force_gpu && !config.static_rule) {
        const auto& stats = controller->GetStats();
        const auto* model = controller->GetModel();
        std::cout << "\n--- Decision Forest & Scheduling Breakdown ---\n";
        if (model && model->IsForestLoaded()) {
            std::cout << "  Active Model:    Random Forest (" << model->NumTrees() << " trees, "
                      << model->TotalNodes() << " nodes, max depth " << model->MaxDepth() << ")\n";
        } else {
            std::cout << "  Active Model:    Analytical Cost Model (Fallback)\n";
        }
        std::cout << "  Decisions Routed:\n";
        std::cout << "    • Guard Rules (Fixed CPU):       " << stats.guard_cpu_tasks << " tasks\n";
        std::cout << "    • Tier-1 Heuristics:              " << (stats.tier1_cpu_tasks + stats.tier1_gpu_tasks)
                  << " tasks (CPU: " << stats.tier1_cpu_tasks << ", GPU: " << stats.tier1_gpu_tasks << ")\n";
        std::cout << "    • Tier-2 Random Forest:           " << (stats.tier2_rf_cpu_tasks + stats.tier2_rf_gpu_tasks)
                  << " tasks (CPU: " << stats.tier2_rf_cpu_tasks << ", GPU: " << stats.tier2_rf_gpu_tasks << ")\n";
        if (stats.dynamic_penalty_shifts > 0) {
            std::cout << "    • Contention Overrides:           " << stats.dynamic_penalty_shifts
                      << " tasks (GPU -> CPU via PCIe/queue penalty)\n";
        }
    }
    std::cout << "===========================\n";
}

int main(int argc, char** argv) {
    std::string trace_path = "benchmarks/traces/mixed_trace.json";
    std::string output_path = "results.csv";

    HeteroSchedConfig config;
    config.cpu_workers = 4;
    config.gpu_stream_pool = 4;
    config.enable_tier1 = true;
    config.enable_tier2 = true;
    config.enable_dynamic_penalties = true;
    config.enable_cross_layer_priority = true;
    config.force_cpu = false;
    config.force_gpu = false;
    config.static_rule = false;
    config.dim_threshold = 512;

    bool has_ablation_flag = false;
    bool has_force_flag = false;
    bool explicit_tier1 = false;
    bool explicit_tier2 = false;
    bool explicit_dynamic = false;
    bool explicit_cross = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--trace") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --trace requires a file path argument.\n";
                return 1;
            }
            trace_path = argv[++i];
        } else if (arg == "--output") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --output requires a file path argument.\n";
                return 1;
            }
            output_path = argv[++i];
        } else if (arg == "--force-cpu") {
            if (config.force_gpu || config.static_rule) {
                std::cerr << "Error: Conflicting scheduling flags specified with --force-cpu.\n";
                return 1;
            }
            config.force_cpu = true;
            has_force_flag = true;
        } else if (arg == "--force-gpu") {
            if (config.force_cpu || config.static_rule) {
                std::cerr << "Error: Conflicting scheduling flags specified with --force-gpu.\n";
                return 1;
            }
            config.force_gpu = true;
            has_force_flag = true;
        } else if (arg == "--static-rule") {
            if (config.force_cpu || config.force_gpu) {
                std::cerr << "Error: Conflicting scheduling flags specified with --static-rule.\n";
                return 1;
            }
            config.static_rule = true;
            has_force_flag = true;
        } else if (arg == "--dim-threshold") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --dim-threshold requires a positive integer argument.\n";
                return 1;
            }
            try {
                long long val = std::stoll(argv[++i]);
                if (val <= 0) {
                    std::cerr << "Error: --dim-threshold must be positive.\n";
                    return 1;
                }
                config.dim_threshold = static_cast<size_t>(val);
            } catch (...) {
                std::cerr << "Error: Invalid argument for --dim-threshold: " << argv[i] << "\n";
                return 1;
            }
        } else if (arg == "--no-scheduling" || arg == "--baseline") {
            has_ablation_flag = true;
            config.enable_tier1 = false;
            config.enable_tier2 = false;
            config.enable_dynamic_penalties = false;
            config.enable_cross_layer_priority = false;
        } else if (arg == "--tier1") {
            has_ablation_flag = true;
            explicit_tier1 = true;
        } else if (arg == "--no-tier1") {
            has_ablation_flag = true;
            explicit_tier1 = false;
        } else if (arg == "--tier2") {
            has_ablation_flag = true;
            explicit_tier2 = true;
        } else if (arg == "--no-tier2") {
            has_ablation_flag = true;
            explicit_tier2 = false;
        } else if (arg == "--dynamic-penalties") {
            has_ablation_flag = true;
            explicit_dynamic = true;
        } else if (arg == "--no-dynamic-penalties") {
            has_ablation_flag = true;
            explicit_dynamic = false;
        } else if (arg == "--cross-layer") {
            has_ablation_flag = true;
            explicit_cross = true;
        } else if (arg == "--no-cross-layer") {
            has_ablation_flag = true;
            explicit_cross = false;
        } else {
            std::cerr << "Error: Unknown argument '" << arg << "'.\n";
            return 1;
        }
    }

    if (has_force_flag && has_ablation_flag) {
        std::cerr << "Error: Forced execution modes (--force-cpu, --force-gpu, --static-rule) cannot be combined with ablation flags.\n";
        return 1;
    }

    if (has_ablation_flag && !config.force_cpu && !config.force_gpu && !config.static_rule) {
        config.enable_tier1 = explicit_tier1;
        config.enable_tier2 = explicit_tier2;
        config.enable_dynamic_penalties = explicit_dynamic;
        config.enable_cross_layer_priority = explicit_cross;
    }

    print_gpu_info();
    gpu_warmup();

    auto trace_entries = load_trace(trace_path);
    if (trace_entries.empty()) {
        std::cerr << "Error: Trace file not found or empty: " << trace_path << "\n";
        return 1;
    }

    std::cout << "Loaded " << trace_entries.size() << " tasks from " << trace_path << "\n";
    std::cout << "Scheduling configuration: " << GetConfigDescription(config) << "\n";

    std::vector<TaskDescriptor> tasks(trace_entries.size());
    for (size_t i = 0; i < trace_entries.size(); ++i) {
        auto& e = trace_entries[i];
        auto& t = tasks[i];
        t = TaskDescriptor{};
        t.task_id = e.id;
        t.type = StringToTaskType(e.type);
        t.priority = e.priority;
        t.dim_m = e.dim_m;
        t.dim_n = e.dim_n;
        t.dim_k = e.dim_k;
        t.bytes_in = e.bytes_in;
        t.bytes_out = e.bytes_out;
        t.estimated_flops = e.estimated_flops;
        bool alloc_ok = allocate_task_buffers(&t);
        if (!alloc_ok) {
            std::cerr << "Warning: Failed to allocate buffers for task ID " << t.task_id << std::endl;
        }
    }

    HeteroSchedRuntime runtime(config);

    auto wall_start = std::chrono::high_resolution_clock::now();

    // Note: The benchmark submission loop below measures synchronous task-completion latency
    // and serializes CPU task submission; it does not measure multicore throughput.
    for (auto& task : tasks) {
        if (!task.host_input || !task.host_output) {
            task.metrics.success = false;
            task.metrics.t_total_latency_us = 0.0;
            continue;
        }
        auto task_start = std::chrono::high_resolution_clock::now();
        runtime.SubmitTask(&task);
        auto task_end = std::chrono::high_resolution_clock::now();
        if (task.metrics.success) {
            task.metrics.t_total_latency_us =
                std::chrono::duration<double, std::micro>(task_end - task_start).count();
        } else {
            task.metrics.t_total_latency_us = 0.0;
        }
    }

    runtime.WaitAll();

    auto wall_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    print_summary(tasks, total_time_ms, config, &runtime.GetController());
    write_results_csv(output_path, tasks, total_time_ms, config);
    std::cout << "Results written to " << output_path << "\n";

    for (auto& t : tasks) {
        free_task_buffers(&t);
    }

    return 0;
}

HeteroSchedRuntime::HeteroSchedRuntime(const HeteroSchedConfig& config)
    : config_(config),
      cpu_scheduler_(config.cpu_workers),
      gpu_executor_(config.gpu_stream_pool),
      controller_(std::make_unique<DecisionForestModel>()),
      gpu_round_robin_(0) {
    controller_.SetTier1Enabled(config.enable_tier1);
    controller_.SetTier2Enabled(config.enable_tier2);
    controller_.SetDynamicPenaltiesEnabled(config.enable_dynamic_penalties);
}

HeteroSchedRuntime::~HeteroSchedRuntime() {
    WaitAll();
}

void HeteroSchedRuntime::SubmitTask(TaskDescriptor* task) {
    if (!task) return;
    if (!task->host_input || !task->host_output) {
        task->metrics.success = false;
        return;
    }

    ExecutionDevice device = ExecutionDevice::DEVICE_CPU;

    if (config_.force_cpu) {
        auto start_decision = std::chrono::high_resolution_clock::now();
        device = ExecutionDevice::DEVICE_CPU;
        auto end_decision = std::chrono::high_resolution_clock::now();
        task->metrics.assigned_device = device;
        task->metrics.t_decision_us =
            std::chrono::duration<double, std::micro>(end_decision - start_decision).count();
    } else if (config_.force_gpu) {
        auto start_decision = std::chrono::high_resolution_clock::now();
        if (task->type == TaskType::QUICKSORT || task->type == TaskType::STRING_TOKEN) {
            device = ExecutionDevice::DEVICE_CPU;
        } else {
            device = ExecutionDevice::DEVICE_GPU;
        }
        auto end_decision = std::chrono::high_resolution_clock::now();
        task->metrics.assigned_device = device;
        task->metrics.t_decision_us =
            std::chrono::duration<double, std::micro>(end_decision - start_decision).count();
    } else if (config_.static_rule) {
        auto start_decision = std::chrono::high_resolution_clock::now();
        if (task->type == TaskType::QUICKSORT || task->type == TaskType::STRING_TOKEN) {
            device = ExecutionDevice::DEVICE_CPU;
        } else if (task->dim_m > config_.dim_threshold) {
            device = ExecutionDevice::DEVICE_GPU;
        } else {
            device = ExecutionDevice::DEVICE_CPU;
        }
        auto end_decision = std::chrono::high_resolution_clock::now();
        task->metrics.assigned_device = device;
        task->metrics.t_decision_us =
            std::chrono::duration<double, std::micro>(end_decision - start_decision).count();
    } else {
        RuntimeState state = telemetry_.Snapshot();
        device = controller_.ScheduleTask(task, state);
    }

    if (device == ExecutionDevice::DEVICE_GPU) {
        ApplyCrossLayerPriority(task, device);
        telemetry_.RecordTransferStart(task->bytes_in + task->bytes_out);
        telemetry_.IncrementGPUQueue();

        size_t stream_idx;
        {
            std::lock_guard<std::mutex> lock(submit_mutex_);
            stream_idx = gpu_round_robin_++;
        }

        gpu_executor_.ExecuteAsync(task, stream_idx);

        telemetry_.DecrementGPUQueue();
        telemetry_.RecordTransferEnd(task->bytes_in + task->bytes_out);
    } else {
        telemetry_.IncrementCPUQueue();
        auto future = cpu_scheduler_.EnqueueWithFuture(task);
        try {
            future.get();
        } catch (...) {
            task->metrics.success = false;
            task->metrics.t_exec_us = 0.0;
            task->metrics.t_total_latency_us = 0.0;
        }
        telemetry_.DecrementCPUQueue();
    }
}

void HeteroSchedRuntime::WaitAll() {
    std::lock_guard<std::mutex> lock(submit_mutex_);
    for (auto& f : pending_futures_) {
        if (f.valid()) f.wait();
    }
    pending_futures_.clear();
}

void HeteroSchedRuntime::ApplyCrossLayerPriority(TaskDescriptor* task, ExecutionDevice device) {
    if (!config_.enable_cross_layer_priority) return;
    if (device != ExecutionDevice::DEVICE_GPU) return;

    if (task->priority >= 2) {
        struct sched_param param;
        param.sched_priority = 1;
        pthread_setschedparam(pthread_self(), SCHED_RR, &param);
    }
}
