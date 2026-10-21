#include "decision_forest_model.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <numeric>

// ─────────────────────────────────────────────────────────────────────
// DecisionTree implementation
// ─────────────────────────────────────────────────────────────────────

int DecisionTree::Predict(const double* features) const {
    if (nodes.empty()) return 0;

    int node_idx = 0;
    while (node_idx >= 0 && node_idx < static_cast<int>(nodes.size())) {
        const auto& node = nodes[node_idx];

        // Leaf node: feature_index == -2 or children are -1
        if (node.feature_index < 0 || node.left_child < 0) {
            if (node.class_counts.size() < 2) return 0;
            return (node.class_counts[1] >= node.class_counts[0]) ? 1 : 0;
        }

        // Internal node: branch on feature threshold
        if (features[node.feature_index] <= node.threshold) {
            node_idx = node.left_child;
        } else {
            node_idx = node.right_child;
        }
    }

    return 0;  // Fallback: CPU
}

double DecisionTree::PredictProbability(const double* features) const {
    if (nodes.empty()) return 0.0;

    int node_idx = 0;
    while (node_idx >= 0 && node_idx < static_cast<int>(nodes.size())) {
        const auto& node = nodes[node_idx];

        if (node.feature_index < 0 || node.left_child < 0) {
            if (node.class_counts.size() < 2) return 0.0;
            double total = node.class_counts[0] + node.class_counts[1];
            if (total <= 0.0) return 0.0;
            return node.class_counts[1] / total;
        }

        if (features[node.feature_index] <= node.threshold) {
            node_idx = node.left_child;
        } else {
            node_idx = node.right_child;
        }
    }

    return 0.0;
}

// ─────────────────────────────────────────────────────────────────────
// Minimal JSON parser (no external dependency)
// Parses the gbm_model.json format produced by export_to_cpp.py
// ─────────────────────────────────────────────────────────────────────

namespace {

// Trim whitespace from string
std::string json_trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end = s.find_last_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

// Parse a JSON array of numbers (ints or doubles) from a substring.
// Expects input like: [1, 2, -2, 3, ...]
std::vector<double> parse_number_array(const std::string& content, size_t& pos) {
    std::vector<double> result;

    // Find opening bracket
    while (pos < content.size() && content[pos] != '[') ++pos;
    ++pos; // skip '['

    std::string num_buf;
    while (pos < content.size()) {
        char c = content[pos];
        if (c == ']') {
            if (!num_buf.empty()) {
                result.push_back(std::stod(json_trim(num_buf)));
                num_buf.clear();
            }
            ++pos;
            break;
        } else if (c == ',') {
            if (!num_buf.empty()) {
                result.push_back(std::stod(json_trim(num_buf)));
                num_buf.clear();
            }
        } else if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
            num_buf += c;
        }
        ++pos;
    }

    return result;
}

// Parse a JSON array of arrays of numbers: [[1,2], [3,4], ...]
std::vector<std::vector<double>> parse_nested_number_array(const std::string& content, size_t& pos) {
    std::vector<std::vector<double>> result;

    // Find opening bracket
    while (pos < content.size() && content[pos] != '[') ++pos;
    ++pos; // skip outer '['

    while (pos < content.size()) {
        char c = content[pos];
        if (c == ']') {
            ++pos;
            break;
        } else if (c == '[') {
            auto inner = parse_number_array(content, pos);
            result.push_back(std::move(inner));
        } else {
            ++pos;
        }
    }

    return result;
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────
// DecisionForestModel implementation
// ─────────────────────────────────────────────────────────────────────

DecisionForestModel::DecisionForestModel()
    : forest_loaded_(false) {
    default_state_.active_pcie_utilization = 0.0;
    default_state_.active_cuda_streams = 0;
    default_state_.pending_gpu_tasks = 0;
    default_state_.pending_cpu_tasks = 0;
    default_state_.cpu_utilization = 0.0;

    // Attempt to auto-load the exported JSON model from the default path.
    // Try multiple search paths so it works from both project root and build/.
    const char* search_paths[] = {
        "ml/models/gbm_model.json",
        "../ml/models/gbm_model.json",
        "../../ml/models/gbm_model.json",
    };
    for (const auto& path : search_paths) {
        if (LoadFromJSON(path)) {
            break;
        }
    }
}

bool DecisionForestModel::LoadFromJSON(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        return false;
    }

    // Read entire file content
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    file.close();

    // Parse the forest structure
    // Expected format:
    // {
    //   "n_estimators": N,
    //   "trees": [
    //     {
    //       "n_nodes": ...,
    //       "feature": [...],
    //       "threshold": [...],
    //       "children_left": [...],
    //       "children_right": [...],
    //       "value": [[c0, c1], ...]
    //     }, ...
    //   ]
    // }

    std::vector<DecisionTree> parsed_trees;

    // Find "n_estimators"
    size_t ne_pos = content.find("\"n_estimators\"");
    if (ne_pos == std::string::npos) {
        std::cerr << "[DecisionForest] JSON missing 'n_estimators' field\n";
        return false;
    }

    // Find the start of the trees array
    size_t trees_pos = content.find("\"trees\"", ne_pos);
    if (trees_pos == std::string::npos) {
        std::cerr << "[DecisionForest] JSON missing 'trees' field\n";
        return false;
    }

    // Find the opening bracket of the trees array
    size_t trees_arr_start = content.find('[', trees_pos);
    if (trees_arr_start == std::string::npos) {
        return false;
    }

    // Parse each tree object
    size_t pos = trees_arr_start + 1;
    while (pos < content.size()) {
        // Find next tree object opening brace
        size_t obj_start = content.find('{', pos);
        if (obj_start == std::string::npos) break;

        // Find the matching closing brace (simple brace counting)
        int brace_depth = 0;
        size_t obj_end = obj_start;
        for (size_t i = obj_start; i < content.size(); ++i) {
            if (content[i] == '{') ++brace_depth;
            else if (content[i] == '}') {
                --brace_depth;
                if (brace_depth == 0) {
                    obj_end = i;
                    break;
                }
            }
        }

        std::string tree_str = content.substr(obj_start, obj_end - obj_start + 1);

        // Parse each field within the tree
        DecisionTree tree;

        // Parse "feature" array
        size_t feat_pos = tree_str.find("\"feature\"");
        std::vector<double> features_raw;
        if (feat_pos != std::string::npos) {
            size_t p = feat_pos;
            features_raw = parse_number_array(tree_str, p);
        }

        // Parse "threshold" array
        size_t thresh_pos = tree_str.find("\"threshold\"");
        std::vector<double> thresholds;
        if (thresh_pos != std::string::npos) {
            size_t p = thresh_pos;
            thresholds = parse_number_array(tree_str, p);
        }

        // Parse "children_left" array
        size_t cl_pos = tree_str.find("\"children_left\"");
        std::vector<double> children_left;
        if (cl_pos != std::string::npos) {
            size_t p = cl_pos;
            children_left = parse_number_array(tree_str, p);
        }

        // Parse "children_right" array
        size_t cr_pos = tree_str.find("\"children_right\"");
        std::vector<double> children_right;
        if (cr_pos != std::string::npos) {
            size_t p = cr_pos;
            children_right = parse_number_array(tree_str, p);
        }

        // Parse "value" nested array
        size_t val_pos = tree_str.find("\"value\"");
        std::vector<std::vector<double>> values;
        if (val_pos != std::string::npos) {
            size_t p = val_pos;
            values = parse_nested_number_array(tree_str, p);
        }

        // Validate array sizes
        size_t n_nodes = features_raw.size();
        if (n_nodes == 0 || thresholds.size() != n_nodes ||
            children_left.size() != n_nodes || children_right.size() != n_nodes ||
            values.size() != n_nodes) {
            std::cerr << "[DecisionForest] Tree " << parsed_trees.size()
                      << " has inconsistent array sizes, skipping\n";
            pos = obj_end + 1;
            continue;
        }

        // Build tree nodes
        tree.nodes.resize(n_nodes);
        for (size_t i = 0; i < n_nodes; ++i) {
            tree.nodes[i].feature_index = static_cast<int>(features_raw[i]);
            tree.nodes[i].threshold = thresholds[i];
            tree.nodes[i].left_child = static_cast<int>(children_left[i]);
            tree.nodes[i].right_child = static_cast<int>(children_right[i]);
            tree.nodes[i].class_counts = values[i];
        }

        parsed_trees.push_back(std::move(tree));
        pos = obj_end + 1;
    }

    if (parsed_trees.empty()) {
        std::cerr << "[DecisionForest] No trees parsed from " << json_path << "\n";
        return false;
    }

    trees_ = std::move(parsed_trees);
    forest_loaded_ = true;
    loaded_path_ = json_path;

    PrintDetailedModelInfo();
    return true;
}

namespace {
int compute_node_depth(const DecisionTree& tree, int node) {
    if (node < 0 || node >= static_cast<int>(tree.nodes.size())) return 0;
    if (tree.nodes[node].left_child < 0) return 1;
    int left_d = compute_node_depth(tree, tree.nodes[node].left_child);
    int right_d = compute_node_depth(tree, tree.nodes[node].right_child);
    return 1 + std::max(left_d, right_d);
}
}  // anonymous namespace

int DecisionForestModel::TotalNodes() const {
    int total = 0;
    for (const auto& tree : trees_) {
        total += static_cast<int>(tree.nodes.size());
    }
    return total;
}

int DecisionForestModel::MaxDepth() const {
    int max_d = 0;
    for (const auto& tree : trees_) {
        int d = compute_node_depth(tree, 0);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

double DecisionForestModel::AvgNodesPerTree() const {
    if (trees_.empty()) return 0.0;
    return static_cast<double>(TotalNodes()) / static_cast<double>(trees_.size());
}

void DecisionForestModel::PrintDetailedModelInfo() const {
    if (!forest_loaded_) {
        std::cout << "[DecisionForest] Status: UNLOADED (Analytical fallback model active)\n";
        return;
    }

    int total_nodes = TotalNodes();
    int max_depth = MaxDepth();
    double avg_nodes = AvgNodesPerTree();

    std::cout << "[DecisionForest] Loaded " << trees_.size()
              << " trees from " << loaded_path_ << "\n";
    std::cout << "================================================================================\n";
    std::cout << "                    HeteroSched Random Forest Model Details                     \n";
    std::cout << "================================================================================\n";
    std::cout << "  Status:           ACTIVE (Trained Random Forest Classifier)\n";
    std::cout << "  Model File:       " << loaded_path_ << "\n";
    std::cout << "  Ensemble Size:    " << trees_.size() << " Decision Trees\n";
    std::cout << "  Total Tree Nodes: " << total_nodes << " nodes (avg: " << avg_nodes << " nodes/tree, max depth: " << max_depth << ")\n";
    std::cout << "  Feature Vector:   10 Dimensions\n";
    std::cout << "                    [0] flops              [5] transfer_time_us\n";
    std::cout << "                    [1] bytes_in           [6] task_type_CONV2D\n";
    std::cout << "                    [2] bytes_out          [7] task_type_QUICKSORT\n";
    std::cout << "                    [3] pcie_load          [8] task_type_SGEMM\n";
    std::cout << "                    [4] gpu_queue_depth    [9] task_type_SPMV\n";
    std::cout << "  Target Output:    Binary Placement (0 = CPU, 1 = GPU) + Confidence Probability\n";
    std::cout << "  Inference Engine: Majority Voting with Cross-Layer Dynamic Contention Biasing\n";
    std::cout << "  Fallback Model:   Analytical Execution Cost Model (Standby)\n";
    std::cout << "================================================================================\n";
}

// ─────────────────────────────────────────────────────────────────────
// Feature extraction
// ─────────────────────────────────────────────────────────────────────

std::array<double, RF_NUM_FEATURES> DecisionForestModel::ExtractFeatures(
    const TaskDescriptor* task, const RuntimeState& state) const {
    // Feature ordering (must match train_model.py after pd.get_dummies):
    //   0: flops
    //   1: bytes_in
    //   2: bytes_out
    //   3: pcie_load
    //   4: gpu_queue_depth
    //   5: transfer_time_us
    //   6: task_type_CONV2D   (one-hot)
    //   7: task_type_QUICKSORT (one-hot)
    //   8: task_type_SGEMM    (one-hot)
    //   9: task_type_SPMV     (one-hot)

    std::array<double, RF_NUM_FEATURES> features{};

    features[0] = static_cast<double>(task->estimated_flops);
    features[1] = static_cast<double>(task->bytes_in);
    features[2] = static_cast<double>(task->bytes_out);
    features[3] = state.active_pcie_utilization;
    features[4] = static_cast<double>(state.pending_gpu_tasks);

    // Compute transfer time estimate: (bytes_in + bytes_out) / 12 GB/s, in microseconds
    double bytes_total = static_cast<double>(task->bytes_in + task->bytes_out);
    features[5] = (bytes_total / 12.0e9) * 1.0e6;

    // One-hot encoding of task type
    features[6] = (task->type == TaskType::CONV2D) ? 1.0 : 0.0;
    features[7] = (task->type == TaskType::QUICKSORT) ? 1.0 : 0.0;
    features[8] = (task->type == TaskType::SGEMM) ? 1.0 : 0.0;
    features[9] = (task->type == TaskType::SPMV) ? 1.0 : 0.0;

    return features;
}

// ─────────────────────────────────────────────────────────────────────
// Random Forest classifier interface
// ─────────────────────────────────────────────────────────────────────

ExecutionDevice DecisionForestModel::Classify(
    const TaskDescriptor* task, const RuntimeState& state) {
    if (!forest_loaded_ || trees_.empty()) {
        // Fallback: use analytical cost comparison
        double cpu_cost = AnalyticalPredictCPU(task);
        double gpu_cost = AnalyticalPredictGPU(task);
        return (cpu_cost <= gpu_cost) ? ExecutionDevice::DEVICE_CPU : ExecutionDevice::DEVICE_GPU;
    }

    auto features = ExtractFeatures(task, state);
    int gpu_votes = 0;
    for (const auto& tree : trees_) {
        gpu_votes += tree.Predict(features.data());
    }

    int n_trees = static_cast<int>(trees_.size());
    return (gpu_votes > n_trees / 2) ? ExecutionDevice::DEVICE_GPU : ExecutionDevice::DEVICE_CPU;
}

double DecisionForestModel::PredictGPUProbability(
    const TaskDescriptor* task, const RuntimeState& state) {
    if (!forest_loaded_ || trees_.empty()) {
        double cpu_cost = AnalyticalPredictCPU(task);
        double gpu_cost = AnalyticalPredictGPU(task);
        return (gpu_cost < cpu_cost) ? 0.8 : 0.2;
    }

    auto features = ExtractFeatures(task, state);
    double prob_sum = 0.0;
    for (const auto& tree : trees_) {
        prob_sum += tree.PredictProbability(features.data());
    }

    return prob_sum / static_cast<double>(trees_.size());
}

// ─────────────────────────────────────────────────────────────────────
// Legacy interface: PredictCPU / PredictGPU (estimated execution time)
// These are still used by the gated controller's cost model.
// ─────────────────────────────────────────────────────────────────────

double DecisionForestModel::PredictCPU(const TaskDescriptor* task) {
    return AnalyticalPredictCPU(task);
}

double DecisionForestModel::PredictGPU(const TaskDescriptor* task) {
    return AnalyticalPredictGPU(task);
}

// ─────────────────────────────────────────────────────────────────────
// Analytical fallback model (original hardcoded formulas)
// ─────────────────────────────────────────────────────────────────────

double DecisionForestModel::AnalyticalPredictCPU(const TaskDescriptor* task) const {
    switch (task->type) {
        case TaskType::QUICKSORT:
            return static_cast<double>(task->dim_m) * 0.05;

        case TaskType::SGEMM: {
            double n = static_cast<double>(task->dim_m);
            return (2.0 * n * n * n) / 3.5e9 * 1e6;
        }

        case TaskType::CONV2D: {
            double pixels = static_cast<double>(task->dim_m * task->dim_n);
            double ksize = static_cast<double>(task->dim_k);
            return (pixels * ksize * ksize * 2.0) / 3.5e9 * 1e6;
        }

        case TaskType::SPMV: {
            double n = static_cast<double>(task->dim_m);
            double nnz = std::min(static_cast<double>(task->dim_n), 10.0);
            return (n * nnz * 2.0) / 3.5e9 * 1e6;
        }

        case TaskType::STRING_TOKEN:
            return static_cast<double>(task->dim_m) * 0.002;
    }
    return 1000.0;
}

double DecisionForestModel::AnalyticalPredictGPU(const TaskDescriptor* task) const {
    switch (task->type) {
        case TaskType::QUICKSORT:
            return 50000.0;

        case TaskType::SGEMM: {
            double n = static_cast<double>(task->dim_m);
            return (2.0 * n * n * n) / 15.0e12 * 1e6;
        }

        case TaskType::CONV2D: {
            double pixels = static_cast<double>(task->dim_m * task->dim_n);
            double ksize = static_cast<double>(task->dim_k);
            return (pixels * ksize * ksize * 2.0) / 15.0e12 * 1e6;
        }

        case TaskType::SPMV: {
            double n = static_cast<double>(task->dim_m);
            double nnz = std::min(static_cast<double>(task->dim_n), 10.0);
            return (n * nnz * 2.0) / 15.0e12 * 1e6;
        }

        case TaskType::STRING_TOKEN:
            return 50000.0;
    }
    return 500.0;
}
