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

    # Force y to be a simple 1D array of floats
    prophet_df = prophet_df.copy()
    prophet_df["y"] = pd.to_numeric(prophet_df["y"], errors='coerce').astype('float64')
    prophet_df = prophet_df.dropna(subset=["y"])
    n_points = len(prophet_df)

    if n_points < 2:
        raise ValueError(f"Video has fewer than 2 valid data points after cleaning")

    # Analyze video's growth pattern to determine forecast behavior
    first_value = float(prophet_df["y"].iloc[0])
    last_value = float(prophet_df["y"].iloc[-1])
    total_growth = last_value - first_value

    # Calculate daily growth rates
    daily_changes = prophet_df["y"].diff().dropna().values
    avg_daily_change = np.mean(daily_changes)
    std_daily_change = np.std(daily_changes) if len(daily_changes) > 1 else avg_daily_change * 0.1

    # Detect trend: accelerating, steady, or decelerating
    if n_points >= 4:
        first_quarter = daily_changes[:len(daily_changes)//2]
        second_quarter = daily_changes[len(daily_changes)//2:]
        early_growth = np.mean(first_quarter) if len(first_quarter) > 0 else avg_daily_change
        late_growth = np.mean(second_quarter) if len(second_quarter) > 0 else avg_daily_change

        # Growth momentum: >1 = accelerating, <1 = decelerating
        if early_growth > 0:
            momentum = late_growth / early_growth
        else:
            momentum = 1.0
    else:
        momentum = 1.0
        early_growth = avg_daily_change
        late_growth = avg_daily_change

    # Use Prophet with moderate flexibility to fit this video's pattern
    model = Prophet(
        growth='linear',
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        changepoint_prior_scale=0.3,  # Moderate - allows fitting the curve
        n_changepoints=max(2, n_points // 2),
        interval_width=0.80
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Get last actual date
    last_actual_date = prophet_df["ds"].max()

    # Apply saturation decay ONLY to future predictions
    for idx in forecast.index:
        forecast_date = forecast.loc[idx, "ds"]

        if forecast_date > last_actual_date:
            days_ahead = (forecast_date - last_actual_date).days

            # Get Prophet's linear projection
            prophet_pred_value = forecast.loc[idx, "yhat"]
            growth_from_last = prophet_pred_value - last_value

            # Apply decay based on video's momentum
            if momentum < 0.8:
                decay = np.exp(-0.20 * days_ahead)
            elif momentum < 1.0:
                decay = np.exp(-0.10 * days_ahead)
            elif momentum > 1.2:
                decay = np.exp(-0.03 * days_ahead)
            else:
                decay = np.exp(-0.07 * days_ahead)

            forecast.loc[idx, "yhat"] = last_value + growth_from_last * decay

            # Scale confidence intervals
            upper_growth = forecast.loc[idx, "yhat_upper"] - last_value
            lower_growth = forecast.loc[idx, "yhat_lower"] - last_value
            forecast.loc[idx, "yhat_upper"] = last_value + upper_growth * decay
            forecast.loc[idx, "yhat_lower"] = last_value + lower_growth * decay

    # Clip negative predictions
    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

    # Metrics from Prophet's fit on historical data (not modified)
    hist_forecast = forecast[forecast["ds"] <= last_actual_date]
    y_true = prophet_df["y"].values
    y_pred = hist_forecast["yhat"].head(len(y_true)).values

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
