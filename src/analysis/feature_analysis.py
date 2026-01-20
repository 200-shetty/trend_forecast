"""
Feature importance analysis for YouTube trend prediction.

Addresses RQ3: What factors drive YouTube virality?
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.inspection import permutation_importance
from scipy import stats
import logging

from src.config import CATEGORY_NAMES, RANDOM_STATE

logger = logging.getLogger(__name__)


def analyze_feature_importance(
    model,
    feature_names: List[str],
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.Series] = None,
    include_permutation: bool = True
) -> pd.DataFrame:
    """
    Comprehensive feature importance analysis.

    Combines:
    - Built-in feature importance (MDI - Mean Decrease in Impurity)
    - Permutation importance (if X, y provided)

    Args:
        model: Trained RandomForest model
        feature_names: List of feature names
        X: Feature DataFrame (for permutation importance)
        y: Target Series (for permutation importance)
        include_permutation: Whether to compute permutation importance

    Returns:
        DataFrame with importance metrics
    """
    # Built-in importance (MDI)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mdi_importance": model.feature_importances_
    })

    # Permutation importance (more reliable but slower)
    if include_permutation and X is not None and y is not None:
        logger.info("Computing permutation importance...")
        perm_importance = permutation_importance(
            model, X, y,
            n_repeats=30,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        importance_df["perm_importance_mean"] = perm_importance.importances_mean
        importance_df["perm_importance_std"] = perm_importance.importances_std

        # Combine scores (average of normalized MDI and permutation)
        mdi_norm = importance_df["mdi_importance"] / importance_df["mdi_importance"].max()
        perm_norm = importance_df["perm_importance_mean"] / importance_df["perm_importance_mean"].max()
        importance_df["combined_importance"] = (mdi_norm + perm_norm) / 2
    else:
        importance_df["combined_importance"] = importance_df["mdi_importance"]

    # Sort and rank
    importance_df = importance_df.sort_values(
        "combined_importance", ascending=False
    ).reset_index(drop=True)

    importance_df["rank"] = range(1, len(importance_df) + 1)
    importance_df["cumulative_importance"] = importance_df["combined_importance"].cumsum()
    importance_df["cumulative_pct"] = (
        importance_df["cumulative_importance"] /
        importance_df["combined_importance"].sum() * 100
    )

    return importance_df


def get_top_features(
    importance_df: pd.DataFrame,
    n: int = 10,
    threshold_pct: Optional[float] = None
) -> pd.DataFrame:
    """
    Get top N features or features explaining threshold% of importance.

    Args:
        importance_df: DataFrame from analyze_feature_importance
        n: Number of top features
        threshold_pct: Cumulative importance threshold (e.g., 80 for 80%)

    Returns:
        Filtered DataFrame
    """
    if threshold_pct is not None:
        return importance_df[importance_df["cumulative_pct"] <= threshold_pct]
    return importance_df.head(n)


def analyze_feature_correlations(
    df: pd.DataFrame,
    target_col: str = "trend_days"
) -> pd.DataFrame:
    """
    Analyze correlations between features and target.

    Args:
        df: Feature DataFrame
        target_col: Target column name

    Returns:
        DataFrame with correlation statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target_col not in numeric_cols:
        raise ValueError(f"Target column {target_col} not found in numeric columns")

    correlations = []
    for col in numeric_cols:
        if col == target_col:
            continue

        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(df[col], df[target_col])

        # Spearman correlation (robust to outliers)
        spearman_r, spearman_p = stats.spearmanr(df[col], df[target_col])

        correlations.append({
            "feature": col,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "abs_pearson": abs(pearson_r),
            "significant": pearson_p < 0.05
        })

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values("abs_pearson", ascending=False).reset_index(drop=True)

    return corr_df


def analyze_category_effects(
    df: pd.DataFrame,
    target_col: str = "trend_days"
) -> pd.DataFrame:
    """
    Analyze how category affects trend duration.

    Args:
        df: Feature DataFrame with categoryId
        target_col: Target column

    Returns:
        Category-level statistics
    """
    if "categoryId" not in df.columns:
        raise ValueError("DataFrame must contain categoryId column")

    category_stats = df.groupby("categoryId")[target_col].agg([
        "count", "mean", "median", "std", "min", "max"
    ]).reset_index()

    # Add category names
    category_stats["category_name"] = category_stats["categoryId"].map(
        CATEGORY_NAMES
    ).fillna("Unknown")

    # Compute effect size (Cohen's d from grand mean)
    grand_mean = df[target_col].mean()
    grand_std = df[target_col].std()

    category_stats["effect_size"] = (
        (category_stats["mean"] - grand_mean) / grand_std
    )

    # Rank by mean trend days
    category_stats = category_stats.sort_values("mean", ascending=False)
    category_stats["rank"] = range(1, len(category_stats) + 1)

    return category_stats


def analyze_temporal_patterns(
    df: pd.DataFrame,
    target_col: str = "trend_days"
) -> Dict:
    """
    Analyze temporal patterns in trending behavior.

    Args:
        df: Feature DataFrame with temporal features
        target_col: Target column

    Returns:
        Dictionary with temporal analysis results
    """
    results = {}

    # Day of week analysis
    if "publish_day_of_week" in df.columns:
        dow_stats = df.groupby("publish_day_of_week")[target_col].agg([
            "count", "mean", "std"
        ]).reset_index()
        dow_stats["day_name"] = dow_stats["publish_day_of_week"].map({
            0: "Monday", 1: "Tuesday", 2: "Wednesday",
            3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
        })
        results["day_of_week"] = dow_stats

        # ANOVA test for day of week effect
        groups = [group[target_col].values for _, group in df.groupby("publish_day_of_week")]
        if len(groups) > 1:
            f_stat, p_value = stats.f_oneway(*groups)
            results["dow_anova"] = {"f_statistic": f_stat, "p_value": p_value}

    # Hour analysis
    if "publish_hour" in df.columns:
        hour_stats = df.groupby("publish_hour")[target_col].agg([
            "count", "mean", "std"
        ]).reset_index()
        results["hour"] = hour_stats

        # Find optimal hours
        top_hours = hour_stats.nlargest(3, "mean")["publish_hour"].tolist()
        results["optimal_hours"] = top_hours

    # Weekend vs weekday
    if "is_weekend_publish" in df.columns:
        weekend_stats = df.groupby("is_weekend_publish")[target_col].agg([
            "count", "mean", "std"
        ]).reset_index()
        weekend_stats["type"] = weekend_stats["is_weekend_publish"].map({
            0: "Weekday", 1: "Weekend"
        })
        results["weekend"] = weekend_stats

        # T-test for weekend effect
        weekday_data = df[df["is_weekend_publish"] == 0][target_col]
        weekend_data = df[df["is_weekend_publish"] == 1][target_col]
        if len(weekday_data) > 0 and len(weekend_data) > 0:
            t_stat, p_value = stats.ttest_ind(weekday_data, weekend_data)
            results["weekend_ttest"] = {"t_statistic": t_stat, "p_value": p_value}

    return results


def generate_feature_report(
    importance_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    category_df: Optional[pd.DataFrame] = None
) -> str:
    """
    Generate a comprehensive feature analysis report.

    Args:
        importance_df: Feature importance DataFrame
        correlation_df: Correlation analysis DataFrame
        category_df: Category effects DataFrame (optional)

    Returns:
        Formatted report string
    """
    report = """
================================================================================
                    FEATURE ANALYSIS REPORT (RQ3: Virality Factors)
================================================================================

TOP 10 MOST IMPORTANT FEATURES:
"""
    top_10 = importance_df.head(10)
    for _, row in top_10.iterrows():
        report += f"  {row['rank']:2d}. {row['feature']:<30} {row['combined_importance']:.4f} "
        report += f"(cumulative: {row['cumulative_pct']:.1f}%)\n"

    report += """
FEATURES REQUIRED FOR 80% IMPORTANCE:
"""
    features_80 = importance_df[importance_df["cumulative_pct"] <= 80]
    report += f"  {len(features_80)} features explain 80% of predictive power\n"

    report += """
TOP CORRELATIONS WITH TREND DURATION:
"""
    top_corr = correlation_df[correlation_df["significant"]].head(10)
    for _, row in top_corr.iterrows():
        report += f"  {row['feature']:<30} r={row['pearson_r']:+.3f} (p={row['pearson_p']:.2e})\n"

    if category_df is not None:
        report += """
CATEGORY EFFECTS:
"""
        for _, row in category_df.head(5).iterrows():
            report += f"  {row['category_name']:<25} mean={row['mean']:.1f} days "
            report += f"(n={row['count']:,})\n"

        report += "\nBottom categories:\n"
        for _, row in category_df.tail(3).iterrows():
            report += f"  {row['category_name']:<25} mean={row['mean']:.1f} days "
            report += f"(n={row['count']:,})\n"

    report += """
================================================================================
KEY FINDINGS:
================================================================================
"""
    # Auto-generate findings
    top_feature = importance_df.iloc[0]["feature"]
    report += f"1. Most important predictor: {top_feature}\n"

    if "early_views" in importance_df["feature"].values:
        early_views_rank = importance_df[importance_df["feature"] == "early_views"]["rank"].values[0]
        report += f"2. Early views importance rank: #{early_views_rank}\n"

    significant_count = len(correlation_df[correlation_df["significant"]])
    report += f"3. {significant_count} features have significant correlation with trend duration\n"

    return report
