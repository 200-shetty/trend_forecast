"""
Load module for YouTube trending data.

Handles saving processed data to various formats with validation.
"""
import pandas as pd
from pathlib import Path
from typing import Optional
import logging
import json

from src.config import DATA_PROCESSED, MODELS_DIR

logger = logging.getLogger(__name__)


def load_features(
    df: pd.DataFrame,
    filename: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Save processed features to parquet format.

    Args:
        df: DataFrame with engineered features
        filename: Output filename (default: features.parquet)
        output_dir: Output directory (default: DATA_PROCESSED)

    Returns:
        Path to saved file
    """
    output_dir = output_dir or DATA_PROCESSED
    filename = filename or "features.parquet"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    df.to_parquet(output_path, engine="pyarrow", index=False)
    logger.info(f"Features saved: {output_path} ({len(df):,} rows)")

    return output_path


def load_model_results(
    results: dict,
    model_name: str,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Save model evaluation results to JSON.

    Args:
        results: Dictionary with model metrics
        model_name: Name of the model (e.g., "random_forest", "prophet")
        output_dir: Output directory (default: MODELS_DIR)

    Returns:
        Path to saved file
    """
    output_dir = output_dir or MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{model_name}_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Model results saved: {output_path}")
    return output_path


def load_forecast(
    forecast_df: pd.DataFrame,
    video_id: str,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Save Prophet forecast to CSV.

    Args:
        forecast_df: DataFrame with forecast results
        video_id: Video ID for filename
        output_dir: Output directory (default: DATA_PROCESSED)

    Returns:
        Path to saved file
    """
    output_dir = output_dir or DATA_PROCESSED
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize video_id for filename
    safe_id = video_id.replace("/", "_").replace("\\", "_")
    output_path = output_dir / f"{safe_id}_forecast.csv"

    forecast_df.to_csv(output_path, index=False)
    logger.info(f"Forecast saved: {output_path}")

    return output_path


def load_quality_report(
    report: dict,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Save data quality report to JSON.

    Args:
        report: Quality report dictionary
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    output_dir = output_dir or DATA_PROCESSED
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "data_quality_report.json"

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Quality report saved: {output_path}")
    return output_path


def read_features(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Read processed features from parquet.

    Args:
        filepath: Path to parquet file (default: looks in DATA_PROCESSED)

    Returns:
        DataFrame with features
    """
    if filepath is None:
        # Find most recent features file
        parquet_files = list(DATA_PROCESSED.glob("*features*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No feature files found in {DATA_PROCESSED}")
        filepath = max(parquet_files, key=lambda p: p.stat().st_mtime)

    df = pd.read_parquet(filepath)
    logger.info(f"Features loaded: {filepath} ({len(df):,} rows)")

    return df
