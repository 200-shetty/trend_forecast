"""
Extract module for YouTube trending data.

Supports single-country and multi-country extraction with data validation.
"""
import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging

from src.config import DATA_RAW

logger = logging.getLogger(__name__)


def extract_raw_data(country: str = "DE") -> pd.DataFrame:
    """
    Extract raw YouTube trending data for a single country.

    Args:
        country: Country code (e.g., "DE", "US", "GB")

    Returns:
        DataFrame with raw trending data

    Raises:
        FileNotFoundError: If CSV file for country doesn't exist
    """
    file_path = DATA_RAW / f"{country}_youtube_trending_data.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file for {country} not found: {file_path}")

    logger.info(f"Extracting data for country: {country}")
    df = pd.read_csv(file_path)
    df["country"] = country

    logger.info(f"Extracted {len(df):,} rows for {country}")
    return df


def extract_all_countries() -> pd.DataFrame:
    """
    Extract and combine data from all available country CSVs.

    Scans the raw data directory for all *_youtube_trending_data.csv files
    and combines them into a single DataFrame with country column.

    Returns:
        Combined DataFrame with all countries' data
    """
    csv_files = list(DATA_RAW.glob("*_youtube_trending_data.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No YouTube trending CSV files found in {DATA_RAW}")

    dfs = []
    for file_path in csv_files:
        country = file_path.stem.replace("_youtube_trending_data", "")
        logger.info(f"Loading {country} from {file_path.name}")

        df = pd.read_csv(file_path)
        df["country"] = country
        dfs.append(df)

        logger.info(f"  - {len(df):,} rows loaded")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total combined: {len(combined):,} rows from {len(dfs)} countries")

    return combined


def get_available_countries() -> List[str]:
    """
    Get list of country codes with available data.

    Returns:
        List of country codes (e.g., ["DE", "US", "GB"])
    """
    csv_files = list(DATA_RAW.glob("*_youtube_trending_data.csv"))
    countries = [f.stem.replace("_youtube_trending_data", "") for f in csv_files]
    return sorted(countries)


def validate_raw_schema(df: pd.DataFrame) -> dict:
    """
    Validate that raw data has expected columns.

    Args:
        df: Raw DataFrame to validate

    Returns:
        Dictionary with validation results
    """
    required_columns = [
        "video_id", "title", "publishedAt", "channelId", "channelTitle",
        "categoryId", "trending_date", "view_count", "likes", "comment_count"
    ]

    optional_columns = [
        "tags", "thumbnail_link", "comments_disabled", "ratings_disabled",
        "description", "dislikes"
    ]

    missing_required = [col for col in required_columns if col not in df.columns]
    present_optional = [col for col in optional_columns if col in df.columns]

    validation = {
        "valid": len(missing_required) == 0,
        "missing_required": missing_required,
        "present_optional": present_optional,
        "total_columns": len(df.columns),
        "total_rows": len(df)
    }

    if not validation["valid"]:
        logger.error(f"Schema validation failed. Missing: {missing_required}")
    else:
        logger.info("Schema validation passed")

    return validation
