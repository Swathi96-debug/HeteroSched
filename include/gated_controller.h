#ifndef GATED_CONTROLLER_H
#define GATED_CONTROLLER_H

#include "common.h"
#include <memory>
#include <chrono>

class DecisionForestModel;

struct ControllerStats {
    size_t total_tasks = 0;
    size_t guard_cpu_tasks = 0;
    size_t tier1_cpu_tasks = 0;
    size_t tier1_gpu_tasks = 0;
    size_t tier2_rf_cpu_tasks = 0;
    size_t tier2_rf_gpu_tasks = 0;
    size_t dynamic_penalty_shifts = 0;
    size_t baseline_tasks = 0;
};

class GatedController {
public:
    explicit GatedController(std::unique_ptr<DecisionForestModel> model);

    ExecutionDevice ScheduleTask(TaskDescriptor* task, const RuntimeState& state);

    void SetTier1Enabled(bool enabled) { tier1_enabled_ = enabled; }
    void SetTier2Enabled(bool enabled) { tier2_enabled_ = enabled; }
    void SetDynamicPenaltiesEnabled(bool enabled) { dynamic_penalties_enabled_ = enabled; }

    const ControllerStats& GetStats() const { return stats_; }
    void ResetStats() { stats_ = ControllerStats{}; }

    const DecisionForestModel* GetModel() const { return model_.get(); }

private:
    void record_decision_overhead(TaskDescriptor* task,
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time);

    std::unique_ptr<DecisionForestModel> model_;
    bool tier1_enabled_ = true;
    bool tier2_enabled_ = true;
    bool dynamic_penalties_enabled_ = true;
    ControllerStats stats_;
};

#endif
