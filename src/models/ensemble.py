"""
Ensemble model combining RandomForest and Prophet.

Novel Contribution (RQ2): RF-predicted trend duration improves Prophet forecasting.

Pipeline:
1. RandomForest predicts trend_days from early engagement features
2. Predicted trend_days becomes an external regressor in Prophet
3. Prophet forecasts daily views informed by expected trend duration
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from src.models.train import RandomForestTrendPredictor
from src.models.prophet_model import ProphetForecast, prepare_prophet_data
from src.config import PROPHET_FORECAST_DAYS, RANDOM_STATE

logger = logging.getLogger(__name__)


class EnsembleForecaster:
    """
    Ensemble model that uses RF predictions to enhance Prophet forecasts.

    The key insight: Early engagement patterns (captured by RF) contain
    information about how long a video will trend, which can improve
    time series forecasting of views.
    """

    def __init__(self):
        self.rf_model: Optional[RandomForestTrendPredictor] = None
        self.prophet_models: Dict[str, ProphetForecast] = {}
        self.rf_features = None
        self.is_fitted = False

    def fit_rf(
        self,
        features_df: pd.DataFrame,
        target_col: str = "trend_days"
    ) -> Dict:
        """
        Train the RandomForest component.

        Args:
            features_df: DataFrame with ML features (from build_features(for_ml=True))
            target_col: Target column name

        Returns:
            Training metrics
        """
        self.rf_model = RandomForestTrendPredictor()
        metrics = self.rf_model.fit(features_df, target_col=target_col)
        self.rf_features = self.rf_model.feature_names

        logger.info("RandomForest component trained")
        return metrics

    def predict_trend_days(
        self,
        features_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict trend duration for videos.

        Args:
            features_df: DataFrame with early engagement features

        Returns:
            Array of predicted trend_days
        """
        if self.rf_model is None:
            raise RuntimeError("RF model must be fitted first")

        # Ensure features match training
        X = features_df[self.rf_features] if self.rf_features else features_df
        return self.rf_model.predict(X)

    def fit_prophet_with_rf(
        self,
        raw_df: pd.DataFrame,
        video_id: str,
        predicted_trend_days: float,
        date_col: str = "trending_date",
        target_col: str = "views"
    ) -> ProphetForecast:
        """
        Fit Prophet model with RF-predicted trend_days as regressor.

        Args:
            raw_df: Raw trending data
            video_id: Video to forecast
            predicted_trend_days: RF prediction for this video
            date_col: Date column name
            target_col: Target column name

        Returns:
            Fitted ProphetForecast model
        """
        # Prepare time series data
        prophet_df = prepare_prophet_data(raw_df, video_id, date_col, target_col)

        # Add RF prediction as constant regressor
        prophet_df["predicted_trend_days"] = predicted_trend_days

        # Create Prophet with regressor
        model = ProphetForecast()
        model.add_regressor("predicted_trend_days")
        model.fit(prophet_df)

        self.prophet_models[video_id] = model
        self.is_fitted = True

        return model

    def forecast_video(
        self,
        raw_df: pd.DataFrame,
        features_df: pd.DataFrame,
        video_id: str,
        periods: int = PROPHET_FORECAST_DAYS
    ) -> Dict:
        """
        Complete ensemble forecast for a single video.

        Args:
            raw_df: Raw trending data
            features_df: Feature DataFrame (with video_id column)
            video_id: Video to forecast
            periods: Forecast horizon

        Returns:
            Dictionary with forecast, metrics, and comparison
        """
        # Get video features
        video_features = features_df[features_df["video_id"] == video_id]

        if len(video_features) == 0:
            raise ValueError(f"Video {video_id} not found in features")

        # Prepare features for RF (drop non-numeric columns)
        rf_input = video_features.drop(
            columns=["video_id", "trending_date", "title", "channelTitle", "country"],
            errors="ignore"
        )

        # Get RF prediction
        if "trend_days" in rf_input.columns:
            actual_trend_days = video_features["trend_days"].values[0]
            rf_input = rf_input.drop(columns=["trend_days"])
        else:
            actual_trend_days = None

        predicted_trend_days = self.predict_trend_days(rf_input)[0]

        # Fit enhanced Prophet
        prophet_model = self.fit_prophet_with_rf(
            raw_df, video_id, predicted_trend_days
        )

        # Create future regressor values
        future_df = prophet_model.model.make_future_dataframe(periods=periods)
        future_df["predicted_trend_days"] = predicted_trend_days

        # Generate forecast
        forecast = prophet_model.model.predict(future_df)

        # Evaluate
        prophet_df = prepare_prophet_data(raw_df, video_id)
        metrics = prophet_model.evaluate(prophet_df)

        return {
            "video_id": video_id,
            "forecast": forecast,
            "metrics": metrics,
            "rf_predicted_trend_days": predicted_trend_days,
            "actual_trend_days": actual_trend_days,
            "forecast_horizon": periods
        }

    def compare_models(
        self,
        raw_df: pd.DataFrame,
        features_df: pd.DataFrame,
        video_id: str,
        periods: int = PROPHET_FORECAST_DAYS
    ) -> Dict:
        """
        Compare ensemble vs baseline Prophet.

        This directly tests RQ2: Does RF enhance Prophet?

        Args:
            raw_df: Raw trending data
            features_df: Feature DataFrame
            video_id: Video to test
            periods: Forecast horizon

        Returns:
            Comparison metrics
        """
        # Prepare data
        prophet_df = prepare_prophet_data(raw_df, video_id)

        if len(prophet_df) < 5:
            logger.warning(f"Video {video_id} has only {len(prophet_df)} points, "
                          "comparison may be unreliable")

        # === Baseline Prophet ===
        baseline_model = ProphetForecast()
        baseline_model.fit(prophet_df)
        baseline_forecast = baseline_model.predict(periods=periods)
        baseline_metrics = baseline_model.evaluate(prophet_df)

        # === Ensemble (RF + Prophet) ===
        ensemble_result = self.forecast_video(
            raw_df, features_df, video_id, periods
        )
        ensemble_metrics = ensemble_result["metrics"]

        # === Calculate Improvement ===
        improvement = {
            "mae_improvement": baseline_metrics["MAE"] - ensemble_metrics["MAE"],
            "mae_improvement_pct": (
                (baseline_metrics["MAE"] - ensemble_metrics["MAE"]) /
                baseline_metrics["MAE"] * 100
            ) if baseline_metrics["MAE"] > 0 else 0,
            "rmse_improvement": baseline_metrics["RMSE"] - ensemble_metrics["RMSE"],
            "r2_improvement": ensemble_metrics["R2"] - baseline_metrics["R2"],
        }

        return {
            "video_id": video_id,
            "baseline": {
                "model": "Prophet (baseline)",
                "metrics": baseline_metrics,
                "forecast": baseline_forecast
            },
            "ensemble": {
                "model": "RF + Prophet (ensemble)",
                "metrics": ensemble_metrics,
                "forecast": ensemble_result["forecast"],
                "rf_prediction": ensemble_result["rf_predicted_trend_days"]
            },
            "improvement": improvement,
            "ensemble_better": improvement["mae_improvement"] > 0
        }


def run_ensemble_experiment(
    raw_df: pd.DataFrame,
    ml_features: pd.DataFrame,
    display_features: pd.DataFrame,
    n_videos: int = 10,
    min_data_points: int = 5
) -> Dict:
    """
    Run full ensemble experiment across multiple videos.

    Tests RQ2: Does RF-informed Prophet outperform baseline Prophet?

    Args:
        raw_df: Raw trending data
        ml_features: Features for RF (numeric only)
        display_features: Features with video_id
        n_videos: Number of videos to test
        min_data_points: Minimum data points required

    Returns:
        Experiment results with statistical summary
    """
    # Train RF on all data
    ensemble = EnsembleForecaster()
    rf_metrics = ensemble.fit_rf(ml_features)

    # Get feature importance
    feature_importance = ensemble.rf_model.get_feature_importance()

    # Select videos with sufficient data
    video_counts = raw_df.groupby("video_id").size()
    eligible_videos = video_counts[video_counts >= min_data_points].index.tolist()

    if len(eligible_videos) < n_videos:
        n_videos = len(eligible_videos)
        logger.warning(f"Only {n_videos} videos have >= {min_data_points} points")

    # Sample videos
    np.random.seed(RANDOM_STATE)
    test_videos = np.random.choice(eligible_videos, size=n_videos, replace=False)

    # Run comparisons
    comparisons = []
    for video_id in test_videos:
        try:
            comparison = ensemble.compare_models(
                raw_df, display_features, video_id
            )
            comparisons.append(comparison)
        except Exception as e:
            logger.warning(f"Failed for video {video_id}: {e}")

    # Aggregate results
    if not comparisons:
        return {"error": "No successful comparisons"}

    ensemble_wins = sum(1 for c in comparisons if c["ensemble_better"])

    mae_improvements = [c["improvement"]["mae_improvement"] for c in comparisons]
    mae_improvement_pcts = [c["improvement"]["mae_improvement_pct"] for c in comparisons]

    results = {
        "rf_metrics": rf_metrics,
        "feature_importance": feature_importance,
        "n_videos_tested": len(comparisons),
        "ensemble_wins": ensemble_wins,
        "ensemble_win_rate": ensemble_wins / len(comparisons),
        "avg_mae_improvement": np.mean(mae_improvements),
        "avg_mae_improvement_pct": np.mean(mae_improvement_pcts),
        "std_mae_improvement": np.std(mae_improvements),
        "comparisons": comparisons
    }

    logger.info(f"Ensemble won {ensemble_wins}/{len(comparisons)} comparisons "
               f"({results['ensemble_win_rate']:.1%})")
    logger.info(f"Average MAE improvement: {results['avg_mae_improvement']:.2f} "
               f"({results['avg_mae_improvement_pct']:.1f}%)")

    return results
