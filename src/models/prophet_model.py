import logging

import pandas as pd
from prophet import Prophet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProphetForecast:
    def __init__(self, daily_seasonality=True):
        self.model = Prophet(daily_seasonality=daily_seasonality)
        self.fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = "trend_days"):
        ts = df.rename(columns={target_col: "y", "ds": "ds"})
        logger.info(f"Fitting Prophet model on {len(ts)} records")
        self.model.fit(ts)
        self.fitted = True

    def predict(self, periods: int = 7) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast

    def evaluate(self, df: pd.DataFrame, target_col: str = "trend_days") -> dict:
        ts = df.rename(columns={target_col: "y", "ds": "ds"})
        pred = self.model.predict(ts)
        y_true = ts["y"].values
        y_pred = pred["yhat"].values
        from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": root_mean_squared_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }
