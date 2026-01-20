import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import logging

logger = logging.getLogger(__name__)


def evaluate_prophet_global(
    clean_df: pd.DataFrame,
    test_ratio: float = 0.2,
    cv_folds: int = 5
) -> dict:
    """
    Evaluate Prophet on predicting trend_days (same target as RandomForest).

    Creates a daily time series of average trend duration and forecasts it.
    Uses cross-validation for fair comparison with RandomForest.
    """

    # Calculate trend_days per video (same logic as build_features)
    trend_bounds = (
        clean_df.groupby("video_id")["trending_date"]
        .agg(first_day="min", last_day="max")
    )
    trend_bounds["trend_days"] = (
        trend_bounds["last_day"] - trend_bounds["first_day"]
    ).dt.days + 1

    # Create daily time series: average trend_days of videos that started trending each day
    df_with_trend = clean_df.merge(
        trend_bounds[["trend_days"]],
        left_on="video_id",
        right_index=True
    )

    # Get first trending date per video
    first_trending = df_with_trend.groupby("video_id").agg(
        first_date=("trending_date", "min"),
        trend_days=("trend_days", "first")
    ).reset_index()

    # Aggregate by date: mean trend_days of videos starting that day
    daily_ts = (
        first_trending
        .groupby("first_date")
        .agg(
            y=("trend_days", "mean"),
            n_videos=("video_id", "count")
        )
        .reset_index()
        .rename(columns={"first_date": "ds"})
        .sort_values("ds")
    )

    # Filter to dates with enough videos for stable estimates
    daily_ts = daily_ts[daily_ts["n_videos"] >= 5].copy()

    if len(daily_ts) < 100:
        logger.warning(f"Only {len(daily_ts)} days with enough data, results may be unstable")

    # Cross-validation to match RandomForest evaluation
    kfold = KFold(n_splits=cv_folds, shuffle=False)  # Time series: no shuffle

    mae_scores = []
    rmse_scores = []
    r2_scores = []

    daily_ts_reset = daily_ts.reset_index(drop=True)

    for fold, (train_idx, test_idx) in enumerate(kfold.split(daily_ts_reset)):
        train_df = daily_ts_reset.iloc[train_idx][["ds", "y"]]
        test_df = daily_ts_reset.iloc[test_idx][["ds", "y"]]

        if len(train_df) < 30 or len(test_df) < 10:
            continue

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative'
        )
        model.fit(train_df)

        forecast = model.predict(test_df[["ds"]])

        y_true = test_df["y"].values
        y_pred = forecast["yhat"].values

        # Clip predictions to valid range (1-30 days typically)
        y_pred = np.clip(y_pred, 1, 30)

        mae_scores.append(mean_absolute_error(y_true, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2_scores.append(r2_score(y_true, y_pred))

    # Also do train/test split evaluation
    split_idx = int(len(daily_ts) * (1 - test_ratio))
    train_df = daily_ts.iloc[:split_idx][["ds", "y"]]
    test_df = daily_ts.iloc[split_idx:][["ds", "y"]]

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='multiplicative'
    )
    model.fit(train_df)

    forecast = model.predict(test_df[["ds"]])
    y_true = test_df["y"].values
    y_pred = np.clip(forecast["yhat"].values, 1, 30)

    test_mae = mean_absolute_error(y_true, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    test_r2 = r2_score(y_true, y_pred)
    test_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "model": "Prophet",
        "target": "trend_days",
        "n_train_days": len(train_df),
        "n_test_days": len(test_df),
        "cv_mae_mean": np.mean(mae_scores) if mae_scores else test_mae,
        "cv_mae_std": np.std(mae_scores) if mae_scores else 0,
        "cv_r2_mean": np.mean(r2_scores) if r2_scores else test_r2,
        "cv_r2_std": np.std(r2_scores) if r2_scores else 0,
        "cv_folds": len(mae_scores),
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "test_mape": test_mape
    }
