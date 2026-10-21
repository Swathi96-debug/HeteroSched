#include <atomic>
#include "gated_controller.h"
#include "decision_forest_model.h"
#include <algorithm>

GatedController::GatedController(std::unique_ptr<DecisionForestModel> model)
    : model_(std::move(model)) {}

ExecutionDevice GatedController::ScheduleTask(TaskDescriptor* task, const RuntimeState& state) {
    auto start_time = std::chrono::high_resolution_clock::now();
    stats_.total_tasks++;

    if (task->type == TaskType::QUICKSORT || task->type == TaskType::STRING_TOKEN) {
        stats_.guard_cpu_tasks++;
        task->metrics.assigned_device = ExecutionDevice::DEVICE_CPU;
        record_decision_overhead(task, start_time);
        return ExecutionDevice::DEVICE_CPU;
    }

    if (tier1_enabled_) {
        double operational_intensity = static_cast<double>(task->estimated_flops) /
                                       (task->bytes_in + task->bytes_out + 1.0);

        if (task->estimated_flops < 100000ULL) {
            stats_.tier1_cpu_tasks++;
            task->metrics.assigned_device = ExecutionDevice::DEVICE_CPU;
            record_decision_overhead(task, start_time);
            return ExecutionDevice::DEVICE_CPU;
        }

        if (operational_intensity < 0.5 && (task->bytes_in + task->bytes_out) > 10 * 1024 * 1024) {
            stats_.tier1_cpu_tasks++;
            task->metrics.assigned_device = ExecutionDevice::DEVICE_CPU;
            record_decision_overhead(task, start_time);
            return ExecutionDevice::DEVICE_CPU;
        }

        if (task->estimated_flops > 1000000000ULL && state.active_pcie_utilization < 0.70) {
            stats_.tier1_gpu_tasks++;
            task->metrics.assigned_device = ExecutionDevice::DEVICE_GPU;
            record_decision_overhead(task, start_time);
            return ExecutionDevice::DEVICE_GPU;
        }
    }

    if (tier2_enabled_ && model_) {
        ExecutionDevice decision;

        // If the random forest is loaded, use it directly for classification.
        // The forest already encodes runtime-aware features (pcie_load, queue depth,
        // transfer time) so it provides contention-aware placement decisions.
        if (model_->IsForestLoaded()) {
            // Get the forest's raw classification and confidence
            double raw_prob = model_->PredictGPUProbability(task, state);
            double gpu_prob = raw_prob;

            if (dynamic_penalties_enabled_) {
                // Apply dynamic penalty adjustments to the forest's probability.
                // When PCIe is saturated or GPU queue is deep, bias towards CPU
                // even if the forest leans GPU. This provides an additional
                // contention-awareness layer on top of the trained features.
                double pcie_penalty = std::max(0.0, state.active_pcie_utilization - 0.70) * 2.5;
                double queue_penalty = (static_cast<double>(state.pending_gpu_tasks) / 32.0) *
                                       (static_cast<double>(state.pending_gpu_tasks) / 32.0) * 1.5;
                double total_penalty = pcie_penalty + queue_penalty * 0.1;

                // Shift the GPU probability down by the penalty
                gpu_prob = std::max(0.0, gpu_prob - total_penalty * 0.15);
            }

            bool raw_favor_gpu = (raw_prob > 0.5);
            bool final_favor_gpu = (gpu_prob > 0.5);

            if (raw_favor_gpu && !final_favor_gpu) {
                stats_.dynamic_penalty_shifts++;
            }

            if (final_favor_gpu) {
                stats_.tier2_rf_gpu_tasks++;
                decision = ExecutionDevice::DEVICE_GPU;
            } else {
                stats_.tier2_rf_cpu_tasks++;
                decision = ExecutionDevice::DEVICE_CPU;
            }
        } else {
            // Fallback: use the analytical cost model (original behavior)
            double alpha = 1.0;
            double beta = 1.0;

            if (dynamic_penalties_enabled_) {
                alpha = 1.0 + 2.5 * std::max(0.0, state.active_pcie_utilization - 0.70);
                beta = 1.0 + 1.5 * (static_cast<double>(state.pending_gpu_tasks) / 32.0)
                           * (static_cast<double>(state.pending_gpu_tasks) / 32.0);
            }

            double pred_cpu_exec = model_->PredictCPU(task);
            double pred_gpu_exec = model_->PredictGPU(task);
            double pred_transfer = static_cast<double>(task->bytes_in + task->bytes_out) /
                                   (12.0 * 1024 * 1024 * 1024 / 1e6);

            double cost_cpu = pred_cpu_exec + (state.pending_cpu_tasks * 200.0);
            double cost_gpu = pred_gpu_exec + (alpha * pred_transfer) +
                              (beta * (state.pending_gpu_tasks * 150.0));

            if (cost_cpu < cost_gpu) {
                stats_.tier2_rf_cpu_tasks++;
                decision = ExecutionDevice::DEVICE_CPU;
            } else {
                stats_.tier2_rf_gpu_tasks++;
                decision = ExecutionDevice::DEVICE_GPU;
            }
        }

        task->metrics.assigned_device = decision;
        record_decision_overhead(task, start_time);
        return decision;
    }

    if (tier1_enabled_ && !tier2_enabled_) {
        stats_.tier1_cpu_tasks++;
        task->metrics.assigned_device = ExecutionDevice::DEVICE_CPU;
        record_decision_overhead(task, start_time);
        return ExecutionDevice::DEVICE_CPU;
    }

    // Baseline (both Tier 1 and Tier 2 disabled): Round-Robin between CPU and GPU
    stats_.baseline_tasks++;
    static std::atomic<size_t> round_robin_counter{0};
    ExecutionDevice decision = (round_robin_counter.fetch_add(1) % 2 == 0)
        ? ExecutionDevice::DEVICE_CPU
        : ExecutionDevice::DEVICE_GPU;
    task->metrics.assigned_device = decision;
    record_decision_overhead(task, start_time);
    return decision;
}

void GatedController::record_decision_overhead(TaskDescriptor* task,
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time) {
    auto end_time = std::chrono::high_resolution_clock::now();
    task->metrics.t_decision_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();
}
