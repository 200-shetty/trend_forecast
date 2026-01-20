"""
Transform module for YouTube trending data.

Handles data cleaning, validation, and quality checks.
"""
import pandas as pd
import numpy as np
from typing import Tuple
import logging

from src.config import MIN_VIEWS_THRESHOLD, MAX_VIEWS_THRESHOLD

logger = logging.getLogger(__name__)


def transform_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw YouTube trending data with cleaning and standardization.

    Performs:
    - Column renaming for consistency
    - DateTime parsing and timezone handling
    - Missing value removal
    - Deduplication
    - Sorting for reproducibility

    Args:
        df: Raw DataFrame from extract step

    Returns:
        Cleaned and transformed DataFrame
    """
    df = df.copy()
    initial_rows = len(df)

    # Standardize column names
    df.rename(
        columns={
            "publishedAt": "published_at",
            "view_count": "views"
        },
        inplace=True
    )

    # Parse datetime columns
    df["published_at"] = pd.to_datetime(
        df["published_at"], errors="coerce"
    ).dt.tz_localize(None)

    df["trending_date"] = pd.to_datetime(
        df["trending_date"], errors="coerce"
    ).dt.tz_localize(None)

    # Remove rows with missing critical fields
    critical_cols = ["video_id", "trending_date", "views", "likes", "comment_count"]
    df.dropna(subset=critical_cols, inplace=True)

    # Remove duplicates (same video on same trending date)
    df.drop_duplicates(subset=["video_id", "trending_date"], inplace=True)

    # Sort for consistent ordering
    df.sort_values(["video_id", "trending_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    rows_removed = initial_rows - len(df)
    logger.info(f"Transform: {initial_rows:,} → {len(df):,} rows ({rows_removed:,} removed)")

    return df


def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Comprehensive data quality validation.

    Checks:
    - Missing values per column
    - Outlier detection for numeric columns
    - Date range validity
    - Negative value detection

    Args:
        df: Transformed DataFrame

    Returns:
        Dictionary with quality metrics and flags
    """
    quality_report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "issues": [],
        "warnings": [],
        "metrics": {}
    }

    # Check missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    quality_report["metrics"]["missing_values"] = missing[missing > 0].to_dict()

    if missing.sum() > 0:
        quality_report["warnings"].append(
            f"Missing values detected in {(missing > 0).sum()} columns"
        )

    # Check numeric column validity
    numeric_cols = ["views", "likes", "comment_count"]
    for col in numeric_cols:
        if col in df.columns:
            negatives = (df[col] < 0).sum()
            if negatives > 0:
                quality_report["issues"].append(
                    f"{negatives:,} negative values in {col}"
                )

            # Outlier detection (beyond reasonable YouTube limits)
            if col == "views":
                extreme = (df[col] > MAX_VIEWS_THRESHOLD).sum()
                if extreme > 0:
                    quality_report["warnings"].append(
                        f"{extreme:,} videos with >2B views (possible data issue)"
                    )

    # Date range checks
    if "trending_date" in df.columns:
        date_range = df["trending_date"].agg(["min", "max"])
        quality_report["metrics"]["date_range"] = {
            "start": str(date_range["min"].date()),
            "end": str(date_range["max"].date()),
            "days_span": (date_range["max"] - date_range["min"]).days
        }

    # Unique counts
    if "video_id" in df.columns:
        quality_report["metrics"]["unique_videos"] = df["video_id"].nunique()

    if "country" in df.columns:
        quality_report["metrics"]["countries"] = df["country"].unique().tolist()

    if "categoryId" in df.columns:
        quality_report["metrics"]["unique_categories"] = df["categoryId"].nunique()

    # Overall quality score
    quality_report["quality_score"] = _calculate_quality_score(quality_report)
    quality_report["valid"] = len(quality_report["issues"]) == 0

    _log_quality_report(quality_report)
    return quality_report


def _calculate_quality_score(report: dict) -> float:
    """Calculate overall quality score (0-100)."""
    score = 100.0

    # Deduct for issues
    score -= len(report["issues"]) * 10

    # Deduct for warnings
    score -= len(report["warnings"]) * 2

    # Deduct for missing values
    missing_cols = len(report["metrics"].get("missing_values", {}))
    score -= missing_cols * 5

    return max(0.0, min(100.0, score))


def _log_quality_report(report: dict):
    """Log quality report summary."""
    logger.info(f"Data Quality Score: {report['quality_score']:.1f}/100")
    logger.info(f"  - Rows: {report['total_rows']:,}")

    if report["metrics"].get("unique_videos"):
        logger.info(f"  - Unique videos: {report['metrics']['unique_videos']:,}")

    if report["metrics"].get("date_range"):
        dr = report["metrics"]["date_range"]
        logger.info(f"  - Date range: {dr['start']} to {dr['end']} ({dr['days_span']} days)")

    if report["issues"]:
        for issue in report["issues"]:
            logger.error(f"  ISSUE: {issue}")

    if report["warnings"]:
        for warning in report["warnings"]:
            logger.warning(f"  WARNING: {warning}")


def remove_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers from a numeric column.

    Args:
        df: DataFrame to process
        column: Column name to check for outliers
        method: 'iqr' (interquartile range) or 'zscore'
        threshold: IQR multiplier or z-score threshold

    Returns:
        DataFrame with outliers removed
    """
    df = df.copy()
    initial_rows = len(df)

    if method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    elif method == "zscore":
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        mask = z_scores < threshold
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")

    df = df[mask]
    removed = initial_rows - len(df)
    logger.info(f"Outlier removal ({column}): {removed:,} rows removed")

    return df
