import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Optional, List
import logging
import warnings

from src.config import PROPHET_FORECAST_DAYS, RANDOM_STATE

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class ProphetForecast:
    def __init__(
        self,
        daily_seasonality: bool = True,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05
    ):
        self.model = Prophet(
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            changepoint_prior_scale=changepoint_prior_scale
        )
        self.fitted = False
        self.regressors = []
        self.training_data = None

    def add_regressor(self, name: str, prior_scale: float = 10.0):
        self.model.add_regressor(name, prior_scale=prior_scale)
        self.regressors.append(name)
        logger.info(f"Added regressor: {name}")

    def fit(
        self,
        df: pd.DataFrame,
        date_col: str = "ds",
        target_col: str = "y"
    ) -> None:
        ts = df.copy()
        if date_col != "ds":
            ts = ts.rename(columns={date_col: "ds"})
        if target_col != "y":
            ts = ts.rename(columns={target_col: "y"})

        ts["ds"] = pd.to_datetime(ts["ds"])

        self.training_data = ts.copy()

        logger.info(f"Fitting Prophet model on {len(ts)} records")
        self.model.fit(ts)
        self.fitted = True

    def predict(
        self,
        periods: int = PROPHET_FORECAST_DAYS,
        include_history: bool = True,
        future_regressors: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate forecasts.

        Args:
            periods: Number of future periods to forecast
            include_history: Include historical predictions
            future_regressors: DataFrame with regressor values for future dates

        Returns:
            DataFrame with forecast results
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")

        future = self.model.make_future_dataframe(periods=periods)

        # Add regressors if needed
        if self.regressors and future_regressors is not None:
            for reg in self.regressors:
                if reg in future_regressors.columns:
                    future = future.merge(
                        future_regressors[["ds", reg]],
                        on="ds",
                        how="left"
                    )
                    future[reg] = future[reg].fillna(future[reg].mean())

        forecast = self.model.predict(future)

        if not include_history:
            forecast = forecast.tail(periods)

        return forecast

    def evaluate(
        self,
        df: pd.DataFrame,
        date_col: str = "ds",
        target_col: str = "y"
    ) -> Dict:
        ts = df.copy()
        if date_col != "ds":
            ts = ts.rename(columns={date_col: "ds"})
        if target_col != "y":
            ts = ts.rename(columns={target_col: "y"})

        ts["ds"] = pd.to_datetime(ts["ds"])

        pred = self.model.predict(ts)
        y_true = ts["y"].values
        y_pred = pred["yhat"].values

        # Calculate coverage probability
        in_bounds = (
            (y_true >= pred["yhat_lower"].values) &
            (y_true <= pred["yhat_upper"].values)
        )
        coverage = in_bounds.mean()

        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred),
            "MAPE": np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100,
            "coverage_80": coverage,
            "n_samples": len(y_true)
        }

    def cross_validate(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        date_col: str = "ds",
        target_col: str = "y"
    ) -> Dict:

        ts = df.copy()
        if date_col != "ds":
            ts = ts.rename(columns={date_col: "ds"})
        if target_col != "y":
            ts = ts.rename(columns={target_col: "y"})

        ts["ds"] = pd.to_datetime(ts["ds"])
        ts = ts.sort_values("ds").reset_index(drop=True)

        tscv = TimeSeriesSplit(n_splits=n_splits)

        mae_scores = []
        r2_scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(ts)):
            train_data = ts.iloc[train_idx]
            test_data = ts.iloc[test_idx]

            if len(train_data) < 10 or len(test_data) < 2:
                continue

            # Create fresh model for each fold
            fold_model = Prophet(
                daily_seasonality=self.model.daily_seasonality,
                weekly_seasonality=self.model.weekly_seasonality,
                yearly_seasonality=self.model.yearly_seasonality
            )

            fold_model.fit(train_data)
            pred = fold_model.predict(test_data)

            mae = mean_absolute_error(test_data["y"], pred["yhat"])
            r2 = r2_score(test_data["y"], pred["yhat"])

            mae_scores.append(mae)
            r2_scores.append(r2)

            logger.debug(f"Fold {fold + 1}: MAE={mae:.2f}, R2={r2:.3f}")

        return {
            "cv_mae_mean": np.mean(mae_scores),
            "cv_mae_std": np.std(mae_scores),
            "cv_r2_mean": np.mean(r2_scores),
            "cv_r2_std": np.std(r2_scores),
            "cv_folds": len(mae_scores)
        }

    def get_components(self) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Model must be fitted first")

        future = self.model.make_future_dataframe(periods=0)
        forecast = self.model.predict(future)

        components = forecast[["ds", "trend", "yhat"]].copy()

        if "daily" in forecast.columns:
            components["daily"] = forecast["daily"]
        if "weekly" in forecast.columns:
            components["weekly"] = forecast["weekly"]

        return components


def prepare_prophet_data(
    df: pd.DataFrame,
    video_id: str,
    date_col: str = "trending_date",
    target_col: str = "views"
) -> pd.DataFrame:

    video_data = df[df["video_id"] == video_id].copy()

    if len(video_data) < 2:
        raise ValueError(f"Video {video_id} has fewer than 2 data points")

    prophet_df = video_data[[date_col, target_col]].copy()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

    return prophet_df


def forecast_video_views(
    df: pd.DataFrame,
    video_id: str,
    periods: int = PROPHET_FORECAST_DAYS,
    date_col: str = "trending_date",
    target_col: str = "views"
) -> Dict:

    # Prepare data
    prophet_df = prepare_prophet_data(df, video_id, date_col, target_col)
    n_points = len(prophet_df)

    # Flatten y values - handle potential nested arrays
    y_values = prophet_df["y"].values
    if hasattr(y_values[0], '__len__') and not isinstance(y_values[0], str):
        # If values are arrays/lists, take first element
        prophet_df["y"] = [float(v[0]) if hasattr(v, '__len__') else float(v) for v in y_values]
    else:
        prophet_df["y"] = prophet_df["y"].astype(float)

    # Use linear growth (simpler, more robust) with good flexibility
    if n_points < 10:
        model = Prophet(
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=0.5,  # High flexibility to follow data
            interval_width=0.80
        )
    elif n_points < 30:
        model = Prophet(
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=0.3,
            interval_width=0.80
        )
    else:
        model = Prophet(
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.2,
            interval_width=0.80
        )

    model.fit(prophet_df)

    # Generate forecast
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Clip negative predictions (views can't be negative)
    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

    # Calculate metrics on training data
    train_forecast = forecast[forecast["ds"].isin(prophet_df["ds"])]
    y_true = prophet_df["y"].values
    y_pred = train_forecast["yhat"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "n_samples": n_points
    }

    return {
        "video_id": video_id,
        "forecast": forecast,
        "metrics": metrics,
        "n_training_points": n_points,
        "forecast_horizon": periods
    }
