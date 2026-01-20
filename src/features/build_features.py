"""
Feature engineering module for YouTube trend prediction.

Builds features from raw trending data for both:
- RandomForest: Predict trend_days (how long a video will stay trending)
- Prophet: Time series forecasting of views

Research-driven features address:
- RQ1: Early engagement → trend longevity
- RQ3: Virality factors (engagement velocity, content features)
- RQ4: Cross-country patterns (country-specific features)
"""
import pandas as pd
import numpy as np
import re
from typing import Optional
import logging

from src.config import EARLY_DAYS, CATEGORY_NAMES

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame, for_ml: bool = False) -> pd.DataFrame:
    """
    Build comprehensive features from raw trending data.

    Args:
        df: Cleaned trending data with multiple rows per video
        for_ml: If True, returns only numeric columns ready for RandomForest.
                If False, includes video_id and trending_date for Prophet/display.

    Returns:
        DataFrame with engineered features
    """
    logger.info(f"Building features from {len(df):,} rows")

    # Calculate trend duration (target variable)
    trend_bounds = (
        df.groupby("video_id")["trending_date"]
        .agg(first_day="min", last_day="max")
    )
    trend_bounds["trend_days"] = (
        trend_bounds["last_day"] - trend_bounds["first_day"]
    ).dt.days + 1

    df = df.join(trend_bounds["trend_days"], on="video_id")

    # Create trend day index for each video
    df["trend_day_index"] = df.groupby("video_id").cumcount() + 1

    # Filter to early days for feature engineering
    early = df[df["trend_day_index"] <= EARLY_DAYS].copy()

    # === BASIC AGGREGATION FEATURES ===
    features = (
        early.groupby("video_id")
        .agg(
            # Early engagement metrics
            early_views=("views", "mean"),
            early_likes=("likes", "mean"),
            early_comments=("comment_count", "mean"),
            max_views=("views", "max"),
            min_views=("views", "min"),

            # Metadata
            categoryId=("categoryId", "first"),
            first_trending=("trending_date", "min"),
            published_at=("published_at", "first"),
            trend_days=("trend_days", "first"),

            # For velocity calculation
            day1_views=("views", "first"),
            day3_views=("views", "last"),
            day1_likes=("likes", "first"),
            day3_likes=("likes", "last"),

            # Content features (if available)
            title=("title", "first"),
            channelTitle=("channelTitle", "first"),

            # Country (if available)
            country=("country", "first") if "country" in df.columns else ("video_id", "first"),
        )
        .reset_index()
    )

    # === TEMPORAL FEATURES ===
    features["days_since_publish"] = (
        features["first_trending"] - features["published_at"]
    ).dt.days

    features["publish_hour"] = features["published_at"].dt.hour
    features["publish_day_of_week"] = features["published_at"].dt.dayofweek
    features["is_weekend_publish"] = features["publish_day_of_week"].isin([5, 6]).astype(int)

    # === ENGAGEMENT RATIO FEATURES ===
    features["likes_view_ratio"] = (
        features["early_likes"] / features["early_views"].replace(0, np.nan)
    ).fillna(0)

    features["comments_view_ratio"] = (
        features["early_comments"] / features["early_views"].replace(0, np.nan)
    ).fillna(0)

    features["engagement_rate"] = (
        (features["early_likes"] + features["early_comments"]) /
        features["early_views"].replace(0, np.nan)
    ).fillna(0)

    # === VELOCITY FEATURES (Early momentum indicators) ===
    features["view_velocity"] = (
        (features["day3_views"] - features["day1_views"]) / max(1, EARLY_DAYS - 1)
    )

    features["like_velocity"] = (
        (features["day3_likes"] - features["day1_likes"]) / max(1, EARLY_DAYS - 1)
    )

    features["view_growth_rate"] = (
        features["day3_views"] / features["day1_views"].replace(0, np.nan)
    ).fillna(1)

    features["views_range"] = features["max_views"] - features["min_views"]

    # Engagement momentum: acceleration of engagement
    features["engagement_momentum"] = (
        features["view_velocity"] * features["likes_view_ratio"]
    )

    # === CONTENT FEATURES ===
    if "title" in features.columns:
        features["title_length"] = features["title"].fillna("").str.len()
        features["has_emoji_title"] = features["title"].fillna("").apply(_has_emoji).astype(int)
        features["title_caps_ratio"] = features["title"].fillna("").apply(_caps_ratio)
        features["title_word_count"] = features["title"].fillna("").str.split().str.len().fillna(0)

    # === CATEGORY FEATURES ===
    features["categoryId"] = pd.to_numeric(
        features["categoryId"], errors="coerce"
    ).fillna(0).astype(int)

    # Category-based statistics (from training data)
    category_stats = _compute_category_stats(df)
    if category_stats is not None:
        features = features.merge(
            category_stats,
            on="categoryId",
            how="left"
        )
        features["category_avg_trend_days"] = features["category_avg_trend_days"].fillna(
            features["trend_days"].mean()
        )
        features["category_competition"] = features["category_competition"].fillna(
            features["category_competition"].median()
        )

    # === CREATOR FEATURES ===
    if "channelTitle" in features.columns:
        creator_stats = _compute_creator_stats(df)
        if creator_stats is not None:
            features = features.merge(
                creator_stats,
                on="channelTitle",
                how="left"
            )
            features["creator_historical_trends"] = features["creator_historical_trends"].fillna(1)
            features["creator_avg_trend_days"] = features["creator_avg_trend_days"].fillna(
                features["trend_days"].mean()
            )

    # === VIRALITY SCORE (Composite) ===
    features["virality_score"] = _compute_virality_score(features)

    # === CLEANUP ===
    # Handle infinities and NaNs
    features.replace([float("inf"), -float("inf")], 0, inplace=True)

    # Drop intermediate columns
    cols_to_drop = ["day1_views", "day3_views", "day1_likes", "day3_likes", "min_views"]
    features.drop(columns=[c for c in cols_to_drop if c in features.columns], inplace=True)

    features.dropna(subset=["trend_days"], inplace=True)

    logger.info(f"Built {len(features):,} feature rows with {len(features.columns)} columns")

    if for_ml:
        return _prepare_for_ml(features)

    return _prepare_for_display(features)


def _prepare_for_ml(features: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for RandomForest (numeric only)."""
    # Columns to drop for ML
    non_numeric_cols = [
        "video_id", "first_trending", "published_at", "trending_date",
        "title", "channelTitle", "country"
    ]

    ml_features = features.drop(
        columns=[c for c in non_numeric_cols if c in features.columns]
    )

    # Ensure all columns are numeric
    for col in ml_features.columns:
        if ml_features[col].dtype == "object":
            ml_features.drop(columns=[col], inplace=True)

    return ml_features


def _prepare_for_display(features: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for display/Prophet (keep identifiers)."""
    display_features = features.copy()

    if "first_trending" in display_features.columns:
        display_features.rename(columns={"first_trending": "trending_date"}, inplace=True)

    if "published_at" in display_features.columns:
        display_features.drop(columns=["published_at"], inplace=True)

    return display_features


def _has_emoji(text: str) -> bool:
    """Check if text contains emoji."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return bool(emoji_pattern.search(str(text)))


def _caps_ratio(text: str) -> float:
    """Calculate ratio of uppercase letters."""
    if not text or len(text) == 0:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _compute_category_stats(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Compute category-level statistics."""
    if "categoryId" not in df.columns:
        return None

    # Get unique videos with their trend_days
    video_trends = df.groupby("video_id").agg(
        categoryId=("categoryId", "first"),
        trend_days=("trend_days", "first") if "trend_days" in df.columns else ("video_id", "count")
    ).reset_index()

    if "trend_days" not in video_trends.columns:
        return None

    category_stats = video_trends.groupby("categoryId").agg(
        category_avg_trend_days=("trend_days", "mean"),
        category_competition=("video_id", "count")  # Number of videos in category
    ).reset_index()

    return category_stats


def _compute_creator_stats(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Compute creator-level historical statistics."""
    if "channelTitle" not in df.columns:
        return None

    # Get unique videos per creator
    video_trends = df.groupby("video_id").agg(
        channelTitle=("channelTitle", "first"),
        trend_days=("trend_days", "first") if "trend_days" in df.columns else ("video_id", "count")
    ).reset_index()

    if "trend_days" not in video_trends.columns:
        return None

    creator_stats = video_trends.groupby("channelTitle").agg(
        creator_historical_trends=("video_id", "count"),
        creator_avg_trend_days=("trend_days", "mean")
    ).reset_index()

    return creator_stats


def _compute_virality_score(features: pd.DataFrame) -> pd.Series:
    """
    Compute composite virality score (0-100).

    Combines:
    - View velocity (normalized)
    - Engagement rate
    - View growth rate
    """
    # Normalize components to 0-1 scale
    def normalize(series):
        min_val, max_val = series.min(), series.max()
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / (max_val - min_val)

    velocity_norm = normalize(features["view_velocity"].clip(lower=0))
    engagement_norm = normalize(features["engagement_rate"].clip(lower=0))
    growth_norm = normalize(features["view_growth_rate"].clip(lower=0, upper=10))

    # Weighted combination
    virality = (
        0.4 * velocity_norm +
        0.3 * engagement_norm +
        0.3 * growth_norm
    ) * 100

    return virality.fillna(50)


def get_feature_descriptions() -> dict:
    """Return descriptions of all features for documentation."""
    return {
        # Target
        "trend_days": "Total days the video appeared in trending (TARGET)",

        # Early engagement
        "early_views": "Mean views during first 3 trending days",
        "early_likes": "Mean likes during first 3 trending days",
        "early_comments": "Mean comments during first 3 trending days",
        "max_views": "Maximum views observed in first 3 days",

        # Engagement ratios
        "likes_view_ratio": "Likes per view (engagement quality)",
        "comments_view_ratio": "Comments per view (audience interaction)",
        "engagement_rate": "(Likes + Comments) / Views",

        # Velocity features
        "view_velocity": "View growth rate per day (day3 - day1) / 2",
        "like_velocity": "Like growth rate per day",
        "view_growth_rate": "day3_views / day1_views (momentum)",
        "views_range": "max_views - min_views (variability)",
        "engagement_momentum": "view_velocity * likes_view_ratio",

        # Temporal features
        "days_since_publish": "Days between upload and first trending",
        "publish_hour": "Hour of day video was published (0-23)",
        "publish_day_of_week": "Day of week published (0=Mon, 6=Sun)",
        "is_weekend_publish": "Published on Saturday or Sunday",

        # Content features
        "title_length": "Character count of video title",
        "has_emoji_title": "Title contains emoji (1/0)",
        "title_caps_ratio": "Ratio of uppercase letters in title",
        "title_word_count": "Number of words in title",

        # Category features
        "categoryId": "YouTube category ID",
        "category_avg_trend_days": "Average trend duration for this category",
        "category_competition": "Number of videos in same category",

        # Creator features
        "creator_historical_trends": "Number of creator's videos that trended",
        "creator_avg_trend_days": "Average trend duration for creator's videos",

        # Composite scores
        "virality_score": "Composite virality indicator (0-100)",
    }
