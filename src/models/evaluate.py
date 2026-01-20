"""
Comprehensive evaluation framework for YouTube trend prediction models.

Provides:
- Multiple regression metrics
- Statistical significance tests
- Model comparison utilities
- Baseline comparisons
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def evaluate(model, X_test, y_test) -> Dict:
    """
    Legacy evaluation function for backward compatibility.

    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test targets

    Returns:
        Dictionary with metrics
    """
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": rmse,
        "R2": r2_score(y_test, preds),
    }


def comprehensive_evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model"
) -> Dict:
    """
    Comprehensive evaluation with multiple metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name for logging

    Returns:
        Dictionary with all metrics
    """
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    pct_errors = np.abs(errors / (y_true + 1e-8)) * 100

    metrics = {
        "model": model_name,
        "n_samples": len(y_true),

        # Standard regression metrics
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAPE": np.mean(pct_errors),

        # Robust metrics
        "median_absolute_error": np.median(abs_errors),
        "max_error": np.max(abs_errors),
        "min_error": np.min(abs_errors),

        # Error distribution
        "error_mean": np.mean(errors),
        "error_std": np.std(errors),
        "error_skew": stats.skew(errors),
        "error_kurtosis": stats.kurtosis(errors),

        # Percentiles
        "error_p25": np.percentile(abs_errors, 25),
        "error_p75": np.percentile(abs_errors, 75),
        "error_p90": np.percentile(abs_errors, 90),
        "error_p95": np.percentile(abs_errors, 95),

        # Correlation
        "pearson_r": stats.pearsonr(y_true, y_pred)[0],
        "spearman_r": stats.spearmanr(y_true, y_pred)[0],
    }

    # Adjusted R2
    n = len(y_true)
    p = 1  # Assume at least 1 predictor
    if n > p + 1:
        metrics["adjusted_R2"] = 1 - (1 - metrics["R2"]) * (n - 1) / (n - p - 1)
    else:
        metrics["adjusted_R2"] = metrics["R2"]

    return metrics


def compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    alpha: float = 0.05
) -> Dict:
    """
    Compare multiple models with statistical significance tests.

    Args:
        y_true: Actual values
        predictions: Dict mapping model names to predictions
        alpha: Significance level

    Returns:
        Comparison results with statistical tests
    """
    results = {
        "metrics": {},
        "pairwise_tests": [],
        "ranking": []
    }

    # Compute metrics for each model
    for name, preds in predictions.items():
        results["metrics"][name] = comprehensive_evaluate(y_true, preds, name)

    # Rank by MAE
    ranked = sorted(
        results["metrics"].items(),
        key=lambda x: x[1]["MAE"]
    )
    results["ranking"] = [name for name, _ in ranked]
    results["best_model"] = results["ranking"][0]

    # Pairwise paired t-tests on absolute errors
    model_names = list(predictions.keys())
    for i, name1 in enumerate(model_names):
        for name2 in model_names[i+1:]:
            errors1 = np.abs(y_true - predictions[name1])
            errors2 = np.abs(y_true - predictions[name2])

            t_stat, p_value = stats.ttest_rel(errors1, errors2)

            test_result = {
                "model1": name1,
                "model2": name2,
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < alpha,
                "better_model": name1 if np.mean(errors1) < np.mean(errors2) else name2
            }
            results["pairwise_tests"].append(test_result)

    return results


def baseline_comparisons(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model"
) -> Dict:
    """
    Compare model against simple baselines.

    Baselines:
    - Mean: Predict the training mean for all
    - Median: Predict the training median for all
    - Last value: Predict y[i-1] for y[i] (for time series)

    Args:
        y_true: Actual values
        y_pred: Model predictions
        model_name: Name of the model

    Returns:
        Comparison with baselines
    """
    # Baseline predictions
    mean_pred = np.full_like(y_true, y_true.mean(), dtype=float)
    median_pred = np.full_like(y_true, np.median(y_true), dtype=float)

    # Last value baseline (shift by 1)
    last_value_pred = np.roll(y_true, 1)
    last_value_pred[0] = y_true[0]

    baselines = {
        model_name: y_pred,
        "Mean Baseline": mean_pred,
        "Median Baseline": median_pred,
        "Last Value Baseline": last_value_pred
    }

    comparison = compare_models(y_true, baselines)

    # Calculate improvement over best baseline
    model_mae = comparison["metrics"][model_name]["MAE"]
    best_baseline_mae = min(
        comparison["metrics"]["Mean Baseline"]["MAE"],
        comparison["metrics"]["Median Baseline"]["MAE"],
        comparison["metrics"]["Last Value Baseline"]["MAE"]
    )

    comparison["improvement_over_baseline"] = {
        "absolute": best_baseline_mae - model_mae,
        "percentage": (best_baseline_mae - model_mae) / best_baseline_mae * 100
        if best_baseline_mae > 0 else 0,
        "beats_all_baselines": model_mae < best_baseline_mae
    }

    return comparison


def calculate_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "MAE",
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a metric.

    Args:
        y_true: Actual values
        y_pred: Predictions
        metric: Metric name ("MAE", "RMSE", "R2")
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples

    Returns:
        (lower_bound, point_estimate, upper_bound)
    """
    n = len(y_true)
    np.random.seed(42)

    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]

        if metric == "MAE":
            value = mean_absolute_error(y_true_boot, y_pred_boot)
        elif metric == "RMSE":
            value = np.sqrt(mean_squared_error(y_true_boot, y_pred_boot))
        elif metric == "R2":
            value = r2_score(y_true_boot, y_pred_boot)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        bootstrap_metrics.append(value)

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_metrics, alpha / 2 * 100)
    upper = np.percentile(bootstrap_metrics, (1 - alpha / 2) * 100)
    point = np.mean(bootstrap_metrics)

    return (lower, point, upper)


def create_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model"
) -> str:
    """
    Create a formatted evaluation report.

    Args:
        y_true: Actual values
        y_pred: Predictions
        model_name: Model name

    Returns:
        Formatted string report
    """
    metrics = comprehensive_evaluate(y_true, y_pred, model_name)
    baseline = baseline_comparisons(y_true, y_pred, model_name)

    # Get confidence intervals
    mae_ci = calculate_confidence_interval(y_true, y_pred, "MAE")
    rmse_ci = calculate_confidence_interval(y_true, y_pred, "RMSE")
    r2_ci = calculate_confidence_interval(y_true, y_pred, "R2")

    report = f"""
================================================================================
                        EVALUATION REPORT: {model_name}
================================================================================

SAMPLE SIZE: {metrics['n_samples']:,}

PRIMARY METRICS (with 95% CI):
  MAE:  {metrics['MAE']:.4f}  [{mae_ci[0]:.4f}, {mae_ci[2]:.4f}]
  RMSE: {metrics['RMSE']:.4f}  [{rmse_ci[0]:.4f}, {rmse_ci[2]:.4f}]
  R²:   {metrics['R2']:.4f}  [{r2_ci[0]:.4f}, {r2_ci[2]:.4f}]
  MAPE: {metrics['MAPE']:.2f}%

ERROR DISTRIBUTION:
  Mean Error:   {metrics['error_mean']:.4f}
  Std Error:    {metrics['error_std']:.4f}
  Median Error: {metrics['median_absolute_error']:.4f}

ERROR PERCENTILES:
  25th: {metrics['error_p25']:.4f}
  75th: {metrics['error_p75']:.4f}
  90th: {metrics['error_p90']:.4f}
  95th: {metrics['error_p95']:.4f}

CORRELATION:
  Pearson:  {metrics['pearson_r']:.4f}
  Spearman: {metrics['spearman_r']:.4f}

BASELINE COMPARISON:
  Best Model: {baseline['best_model']}
  Beats All Baselines: {baseline['improvement_over_baseline']['beats_all_baselines']}
  Improvement: {baseline['improvement_over_baseline']['percentage']:.1f}%

================================================================================
"""
    return report
