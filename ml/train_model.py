"""
HeteroSched Random Forest Training Pipeline
=============================================
Trains two model variants:
  1. Classifier: Predicts optimal device (CPU=0 / GPU=1) for task scheduling.
  2. Regressor pair: Predicts CPU and GPU execution times (microseconds)
     for the cost-model based scheduler.

Outputs:
  - ml/models/rf_scheduler.pkl       (classifier)
  - ml/models/rf_cpu_regressor.pkl   (CPU time regressor)
  - ml/models/rf_gpu_regressor.pkl   (GPU time regressor)
  - ml/models/training_report.txt    (accuracy, feature importance, CV scores)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import joblib
import os
import sys
import time


def load_and_prepare_data(csv_path="ml/dataset.csv"):
    """Load dataset and prepare features/targets."""
    df = pd.read_csv(csv_path)
    df_encoded = pd.get_dummies(df, columns=["task_type"])

    feature_cols = [
        c for c in df_encoded.columns
        if c not in ["optimal_device", "cpu_time_us", "gpu_time_us"]
    ]

    X = df_encoded[feature_cols]
    y_class = df_encoded["optimal_device"]
    y_cpu_time = df_encoded["cpu_time_us"]
    y_gpu_time = df_encoded["gpu_time_us"]

    return X, y_class, y_cpu_time, y_gpu_time, feature_cols


def train_classifier(X_train, X_test, y_train, y_test, feature_cols, report_lines,
                     tune_hyperparams=False):
    """Train the device placement classifier with optional hyperparameter tuning."""
    report_lines.append("=" * 60)
    report_lines.append("RANDOM FOREST CLASSIFIER (Device Placement)")
    report_lines.append("=" * 60)

    if tune_hyperparams:
        report_lines.append("\n[Hyperparameter Tuning] Running GridSearchCV...")
        param_grid = {
            "n_estimators": [10, 20, 50],
            "max_depth": [4, 6, 8, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        clf = grid_search.best_estimator_
        report_lines.append(f"  Best params: {grid_search.best_params_}")
        report_lines.append(f"  Best CV accuracy: {grid_search.best_score_:.4f}")
    else:
        clf = RandomForestClassifier(
            n_estimators=20,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

    # Cross-validation scores
    cv_scores = cross_val_score(
        clf, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
    )
    report_lines.append(f"\n[Cross-Validation] 5-fold accuracy: "
                        f"{cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    report_lines.append(f"  Per-fold: {', '.join(f'{s:.4f}' for s in cv_scores)}")

    # Test set evaluation
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report_lines.append(f"\n[Test Set] Accuracy: {acc:.4f}")
    report_lines.append(f"\n{classification_report(y_test, preds, target_names=['CPU', 'GPU'])}")

    cm = confusion_matrix(y_test, preds)
    report_lines.append("Confusion Matrix:")
    report_lines.append(f"  {'':>10} Pred_CPU  Pred_GPU")
    report_lines.append(f"  {'True_CPU':>10}  {cm[0][0]:>6}    {cm[0][1]:>6}")
    report_lines.append(f"  {'True_GPU':>10}  {cm[1][0]:>6}    {cm[1][1]:>6}")

    # Feature importance
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    report_lines.append("\n[Feature Importance]")
    for rank, idx in enumerate(sorted_idx):
        report_lines.append(f"  {rank + 1}. {feature_cols[idx]:>25s}: {importances[idx]:.4f}")

    # Inference latency benchmark
    import warnings
    single_sample = X_test.iloc[:1]  # Keep as DataFrame to avoid feature name warnings
    n_bench = 10000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        for _ in range(n_bench):
            clf.predict(single_sample)
        elapsed = (time.perf_counter() - start) / n_bench * 1e6
    report_lines.append(f"\n[Inference Latency] {elapsed:.1f} us per prediction "
                        f"(Python sklearn, {n_bench} iterations)")

    # Model stats
    total_nodes = sum(t.tree_.node_count for t in clf.estimators_)
    max_depth_actual = max(t.tree_.max_depth for t in clf.estimators_)
    report_lines.append(f"\n[Model Stats]")
    report_lines.append(f"  Trees: {len(clf.estimators_)}")
    report_lines.append(f"  Total nodes: {total_nodes}")
    report_lines.append(f"  Max tree depth (actual): {max_depth_actual}")
    report_lines.append(f"  Avg nodes per tree: {total_nodes / len(clf.estimators_):.1f}")

    return clf


def train_regressors(X_train, X_test, y_cpu_train, y_cpu_test,
                     y_gpu_train, y_gpu_test, feature_cols, report_lines):
    """Train regression models for CPU and GPU execution time prediction."""
    report_lines.append("\n" + "=" * 60)
    report_lines.append("RANDOM FOREST REGRESSORS (Execution Time Prediction)")
    report_lines.append("=" * 60)

    for name, y_train, y_test in [
        ("CPU Time", y_cpu_train, y_cpu_test),
        ("GPU Time", y_gpu_train, y_gpu_test),
    ]:
        report_lines.append(f"\n--- {name} Regressor ---")

        reg = RandomForestRegressor(
            n_estimators=30,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        reg.fit(X_train, y_train)

        preds = reg.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        report_lines.append(f"  MAE:  {mae:.2f} us")
        report_lines.append(f"  RMSE: {rmse:.2f} us")
        report_lines.append(f"  R²:   {r2:.4f}")

        # Cross-validation R²
        cv_r2 = cross_val_score(reg, X_train, y_train, cv=5, scoring="r2")
        report_lines.append(f"  CV R²: {cv_r2.mean():.4f} (+/- {cv_r2.std() * 2:.4f})")

        # Feature importance
        importances = reg.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:5]
        report_lines.append(f"  Top-5 features:")
        for idx in sorted_idx:
            report_lines.append(f"    {feature_cols[idx]:>25s}: {importances[idx]:.4f}")

        # Save
        model_name = "rf_cpu_regressor" if "CPU" in name else "rf_gpu_regressor"
        save_path = f"ml/models/{model_name}.pkl"
        joblib.dump(reg, save_path)
        report_lines.append(f"  Saved to {save_path}")


def train_and_evaluate(tune_hyperparams=False):
    """Main training entry point."""
    print("=" * 60)
    print(" HeteroSched Random Forest Training Pipeline")
    print("=" * 60)

    X, y_class, y_cpu_time, y_gpu_time, feature_cols = load_and_prepare_data()
    print(f"Dataset: {len(X)} samples, {len(feature_cols)} features")
    print(f"Class distribution: CPU={sum(y_class == 0)}, GPU={sum(y_class == 1)}")

    # Split data (shared split for classifier and regressors)
    X_train, X_test, y_class_train, y_class_test, y_cpu_train, y_cpu_test, \
        y_gpu_train, y_gpu_test = train_test_split(
            X, y_class, y_cpu_time, y_gpu_time,
            test_size=0.2, random_state=42, stratify=y_class,
        )

    report_lines = []
    report_lines.append(f"Training Report - HeteroSched Random Forest")
    report_lines.append(f"Dataset: {len(X)} samples, {len(feature_cols)} features")
    report_lines.append(f"Train/Test split: {len(X_train)} / {len(X_test)}")
    report_lines.append(f"Class balance: CPU={sum(y_class == 0)}, GPU={sum(y_class == 1)}")
    report_lines.append(f"Feature columns: {feature_cols}")
    report_lines.append("")

    # 1. Train classifier
    clf = train_classifier(
        X_train, X_test, y_class_train, y_class_test,
        feature_cols, report_lines, tune_hyperparams,
    )

    # 2. Train regressors
    train_regressors(
        X_train, X_test, y_cpu_train, y_cpu_test,
        y_gpu_train, y_gpu_test, feature_cols, report_lines,
    )

    # Save models
    os.makedirs("ml/models", exist_ok=True)
    joblib.dump(clf, "ml/models/rf_scheduler.pkl")
    print(f"\nClassifier saved to ml/models/rf_scheduler.pkl")

    # Write report
    report_path = "ml/models/training_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Training report written to {report_path}")

    # Print summary
    preds = clf.predict(X_test)
    print(f"\nClassifier accuracy: {accuracy_score(y_class_test, preds):.4f}")
    print(f"Trees: {len(clf.estimators_)}, "
          f"Total nodes: {sum(t.tree_.node_count for t in clf.estimators_)}")


if __name__ == "__main__":
    tune = "--tune" in sys.argv
    if tune:
        print("[INFO] Hyperparameter tuning enabled (this may take a few minutes)")
    train_and_evaluate(tune_hyperparams=tune)
