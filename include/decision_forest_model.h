#ifndef DECISION_FOREST_MODEL_H
#define DECISION_FOREST_MODEL_H

#include "common.h"
#include <vector>
#include <string>
#include <array>

// Number of features expected by the trained random forest.
// [flops, bytes_in, bytes_out, pcie_load, gpu_queue_depth,
//  transfer_time_us, task_type_CONV2D, task_type_QUICKSORT,
//  task_type_SGEMM, task_type_SPMV]
static constexpr int RF_NUM_FEATURES = 10;

// A single decision tree node for runtime JSON-loaded inference.
struct TreeNode {
    int feature_index;      // -2 for leaf nodes
    double threshold;
    int left_child;         // -1 for leaf nodes
    int right_child;        // -1 for leaf nodes
    std::vector<double> class_counts;  // vote counts per class at leaf
};

// A single decision tree composed of nodes.
struct DecisionTree {
    std::vector<TreeNode> nodes;

    // Traverse the tree and return the predicted class (0=CPU, 1=GPU).
    int Predict(const double* features) const;

    // Traverse and return the class probability for the positive class (GPU).
    double PredictProbability(const double* features) const;
};

class DecisionForestModel {
public:
    DecisionForestModel();
    ~DecisionForestModel() = default;

    // Load a random forest from the exported JSON file.
    // Returns true on success. If loading fails, falls back to analytical model.
    bool LoadFromJSON(const std::string& json_path);

    // Original interface: predict estimated execution time on CPU/GPU (microseconds).
    // Uses the analytical fallback model when no forest is loaded.
    double PredictCPU(const TaskDescriptor* task);
    double PredictGPU(const TaskDescriptor* task);

    // Random Forest classifier interface: directly predict the optimal device.
    // Returns DEVICE_GPU if the forest majority-votes GPU, else DEVICE_CPU.
    // Requires RuntimeState for pcie_load and queue depth features.
    ExecutionDevice Classify(const TaskDescriptor* task, const RuntimeState& state);

    // Returns the probability (0.0 to 1.0) that GPU is the optimal device.
    double PredictGPUProbability(const TaskDescriptor* task, const RuntimeState& state);

    // Returns true if a trained forest has been loaded (JSON or compiled).
    bool IsForestLoaded() const { return forest_loaded_; }

    // Returns the number of trees in the loaded forest.
    int NumTrees() const { return static_cast<int>(trees_.size()); }

    // Model statistics
    int TotalNodes() const;
    int MaxDepth() const;
    double AvgNodesPerTree() const;
    const std::string& LoadedModelPath() const { return loaded_path_; }

    // Print a comprehensive, human-readable summary of the loaded Random Forest
    void PrintDetailedModelInfo() const;

private:
    // Extract the 10-element feature vector from task descriptor + runtime state.
    std::array<double, RF_NUM_FEATURES> ExtractFeatures(
        const TaskDescriptor* task, const RuntimeState& state) const;

    // Analytical fallback predictions (the original hardcoded formulas).
    double AnalyticalPredictCPU(const TaskDescriptor* task) const;
    double AnalyticalPredictGPU(const TaskDescriptor* task) const;

    std::vector<DecisionTree> trees_;
    bool forest_loaded_;
    std::string loaded_path_;

    // Default runtime state used when Classify/PredictGPUProbability
    // is called without explicit state (for the PredictCPU/GPU interface).
    RuntimeState default_state_;
};

#endif
