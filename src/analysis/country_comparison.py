"""
Cross-country analysis for YouTube trending patterns.

Addresses RQ4: How do trending patterns differ across countries?
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
import logging

from src.config import CATEGORY_NAMES

logger = logging.getLogger(__name__)


def compare_countries(
    df: pd.DataFrame,
    target_col: str = "trend_days"
) -> Dict:
    """
    Statistical comparison of trending patterns across countries.

    Args:
        df: DataFrame with country column
        target_col: Target column to compare

    Returns:
        Dictionary with comparison results
    """
    if "country" not in df.columns:
        raise ValueError("DataFrame must contain 'country' column")

    countries = df["country"].unique().tolist()

    if len(countries) < 2:
        logger.warning("Only one country in data, comparison not possible")
        return {"single_country": countries[0], "comparison_possible": False}

    results = {
        "countries": countries,
        "comparison_possible": True,
        "country_stats": {},
        "statistical_tests": {}
    }

    # Per-country statistics
    for country in countries:
        country_data = df[df["country"] == country][target_col]
        results["country_stats"][country] = {
            "n_videos": len(country_data),
            "mean": country_data.mean(),
            "median": country_data.median(),
            "std": country_data.std(),
            "min": country_data.min(),
            "max": country_data.max(),
            "q25": country_data.quantile(0.25),
            "q75": country_data.quantile(0.75)
        }

    # Statistical tests
    if len(countries) == 2:
        # Two countries: t-test
        data1 = df[df["country"] == countries[0]][target_col]
        data2 = df[df["country"] == countries[1]][target_col]

        t_stat, p_value = stats.ttest_ind(data1, data2)
        results["statistical_tests"]["ttest"] = {
            "countries": countries,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05
        }

        # Mann-Whitney U test (non-parametric)
        u_stat, p_value_mw = stats.mannwhitneyu(data1, data2, alternative="two-sided")
        results["statistical_tests"]["mannwhitney"] = {
            "u_statistic": u_stat,
            "p_value": p_value_mw,
            "significant": p_value_mw < 0.05
        }

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
        cohens_d = (data1.mean() - data2.mean()) / pooled_std
        results["statistical_tests"]["effect_size"] = {
            "cohens_d": cohens_d,
            "interpretation": _interpret_cohens_d(cohens_d)
        }

    else:
        # Multiple countries: ANOVA
        groups = [df[df["country"] == c][target_col].values for c in countries]
        f_stat, p_value = stats.f_oneway(*groups)
        results["statistical_tests"]["anova"] = {
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < 0.05
        }

        # Kruskal-Wallis test (non-parametric)
        h_stat, p_value_kw = stats.kruskal(*groups)
        results["statistical_tests"]["kruskal"] = {
            "h_statistic": h_stat,
            "p_value": p_value_kw,
            "significant": p_value_kw < 0.05
        }

    # Rank countries by mean trend duration
    ranked = sorted(
        results["country_stats"].items(),
        key=lambda x: x[1]["mean"],
        reverse=True
    )
    results["ranking"] = [c for c, _ in ranked]

    return results


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def analyze_category_distribution(
    df: pd.DataFrame,
    by_country: bool = True
) -> pd.DataFrame:
    """
    Analyze category distribution across countries.

    Args:
        df: DataFrame with categoryId and optionally country
        by_country: Whether to break down by country

    Returns:
        DataFrame with category distribution
    """
    if "categoryId" not in df.columns:
        raise ValueError("DataFrame must contain 'categoryId' column")

    if by_country and "country" in df.columns:
        # Distribution per country
        dist = df.groupby(["country", "categoryId"]).agg(
            count=("video_id", "nunique") if "video_id" in df.columns else ("categoryId", "count")
        ).reset_index()

        # Calculate percentage within each country
        country_totals = dist.groupby("country")["count"].transform("sum")
        dist["percentage"] = dist["count"] / country_totals * 100

    else:
        # Overall distribution
        dist = df.groupby("categoryId").agg(
            count=("video_id", "nunique") if "video_id" in df.columns else ("categoryId", "count")
        ).reset_index()
        dist["percentage"] = dist["count"] / dist["count"].sum() * 100

    # Add category names
    dist["category_name"] = dist["categoryId"].map(CATEGORY_NAMES).fillna("Unknown")

    return dist


def compare_category_preferences(
    df: pd.DataFrame
) -> Dict:
    """
    Compare category preferences across countries using chi-square test.

    Args:
        df: DataFrame with country and categoryId

    Returns:
        Chi-square test results
    """
    if "country" not in df.columns or "categoryId" not in df.columns:
        raise ValueError("DataFrame must contain 'country' and 'categoryId' columns")

    # Create contingency table
    contingency = pd.crosstab(df["country"], df["categoryId"])

    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    # Cramér's V (effect size for chi-square)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    results = {
        "chi2_statistic": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "cramers_v": cramers_v,
        "significant": p_value < 0.05,
        "interpretation": _interpret_cramers_v(cramers_v),
        "contingency_table": contingency
    }

    return results


def _interpret_cramers_v(v: float) -> str:
    """Interpret Cramér's V effect size."""
    if v < 0.1:
        return "negligible association"
    elif v < 0.3:
        return "weak association"
    elif v < 0.5:
        return "moderate association"
    else:
        return "strong association"


def analyze_engagement_by_country(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare engagement metrics across countries.

    Args:
        df: DataFrame with engagement metrics

    Returns:
        Country-level engagement statistics
    """
    if "country" not in df.columns:
        raise ValueError("DataFrame must contain 'country' column")

    engagement_cols = ["early_views", "early_likes", "early_comments",
                      "likes_view_ratio", "engagement_rate", "virality_score"]

    available_cols = [c for c in engagement_cols if c in df.columns]

    if not available_cols:
        logger.warning("No engagement columns found")
        return pd.DataFrame()

    # Aggregate by country
    agg_dict = {col: ["mean", "median", "std"] for col in available_cols}
    agg_dict["video_id"] = "nunique" if "video_id" in df.columns else "count"

    stats_df = df.groupby("country").agg(agg_dict)
    stats_df.columns = ["_".join(col).strip() for col in stats_df.columns]
    stats_df = stats_df.reset_index()

    return stats_df


def generate_country_report(
    comparison_results: Dict,
    category_dist: pd.DataFrame
) -> str:
    """
    Generate cross-country analysis report.

    Args:
        comparison_results: Results from compare_countries
        category_dist: Category distribution DataFrame

    Returns:
        Formatted report string
    """
    report = """
================================================================================
              CROSS-COUNTRY ANALYSIS REPORT (RQ4: Regional Differences)
================================================================================

"""
    if not comparison_results.get("comparison_possible", False):
        report += f"Only one country available: {comparison_results.get('single_country', 'Unknown')}\n"
        report += "Cross-country comparison not possible with current data.\n"
        return report

    # Country statistics
    report += "COUNTRY STATISTICS:\n"
    report += "-" * 60 + "\n"
    for country, stats in comparison_results["country_stats"].items():
        report += f"\n{country}:\n"
        report += f"  Videos: {stats['n_videos']:,}\n"
        report += f"  Mean trend duration: {stats['mean']:.2f} days\n"
        report += f"  Median: {stats['median']:.2f}, Std: {stats['std']:.2f}\n"

    # Statistical tests
    report += "\nSTATISTICAL TESTS:\n"
    report += "-" * 60 + "\n"

    tests = comparison_results.get("statistical_tests", {})
    if "ttest" in tests:
        t = tests["ttest"]
        report += f"Independent t-test: t={t['t_statistic']:.3f}, p={t['p_value']:.4f}\n"
        report += f"  Significant difference: {'Yes' if t['significant'] else 'No'}\n"

    if "effect_size" in tests:
        e = tests["effect_size"]
        report += f"Effect size (Cohen's d): {e['cohens_d']:.3f} ({e['interpretation']})\n"

    if "anova" in tests:
        a = tests["anova"]
        report += f"ANOVA: F={a['f_statistic']:.3f}, p={a['p_value']:.4f}\n"
        report += f"  Significant difference: {'Yes' if a['significant'] else 'No'}\n"

    # Ranking
    report += "\nCOUNTRY RANKING (by mean trend duration):\n"
    report += "-" * 60 + "\n"
    for i, country in enumerate(comparison_results.get("ranking", []), 1):
        mean_days = comparison_results["country_stats"][country]["mean"]
        report += f"  {i}. {country}: {mean_days:.2f} days\n"

    # Category distribution summary
    if not category_dist.empty and "country" in category_dist.columns:
        report += "\nTOP CATEGORIES BY COUNTRY:\n"
        report += "-" * 60 + "\n"
        for country in category_dist["country"].unique():
            country_cats = category_dist[category_dist["country"] == country]
            top_cat = country_cats.nlargest(1, "percentage").iloc[0]
            report += f"  {country}: {top_cat['category_name']} ({top_cat['percentage']:.1f}%)\n"

    report += """
================================================================================
"""
    return report
