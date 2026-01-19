import pandas as pd
from src.config import EARLY_DAYS


def build_features(df: pd.DataFrame, for_ml: bool = False) -> pd.DataFrame:
    """
    Build features from raw trending data.

    Args:
        df: Cleaned trending data with multiple rows per video
        for_ml: If True, returns only numeric columns ready for RandomForest.
                If False, includes video_id and trending_date for Prophet/display.
    """
    trend_bounds = (
        df.groupby("video_id")["trending_date"]
        .agg(first_day="min", last_day="max")
    )

    trend_bounds["trend_days"] = (
        trend_bounds["last_day"] - trend_bounds["first_day"]
    ).dt.days + 1

    df = df.join(trend_bounds["trend_days"], on="video_id")

    df["trend_day_index"] = df.groupby("video_id").cumcount() + 1

    early = df[df["trend_day_index"] <= EARLY_DAYS]

    features = (
        early.groupby("video_id")
        .agg(
            early_views=("views", "mean"),
            early_likes=("likes", "mean"),
            early_comments=("comment_count", "mean"),
            max_views=("views", "max"),
            categoryId=("categoryId", "first"),
            first_trending=("trending_date", "min"),
            published_at=("published_at", "first"),
            trend_days=("trend_days", "first")
        )
        .reset_index()
    )

    features["days_since_publish"] = (
        features["first_trending"] - features["published_at"]
    ).dt.days

    features["likes_view_ratio"] = features["early_likes"] / features["early_views"]
    features["comments_view_ratio"] = features["early_comments"] / features["early_views"]

    # Convert categoryId to numeric
    features["categoryId"] = pd.to_numeric(features["categoryId"], errors="coerce").fillna(0).astype(int)

    features.replace([float("inf"), -float("inf")], 0, inplace=True)
    features.dropna(inplace=True)

    if for_ml:
        # Return only numeric columns for RandomForest
        return features.drop(columns=["video_id", "first_trending", "published_at"])

    # Return full features with identifiers for Prophet/display
    return features.drop(columns=["published_at"]).rename(columns={"first_trending": "trending_date"})
