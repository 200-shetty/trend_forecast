from src.etl.extract import extract_raw_data
from src.etl.transform import transform_raw_data
from src.features.build_features import build_features
from src.models.prophet_model import ProphetForecast
from src.config import RAW_FILE
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(country="DE", video_id=None, forecast_days=7):
    raw = extract_raw_data(country)
    clean = transform_raw_data(raw)
    features = build_features(clean)

    processed_file = f"data/processed/{country}_features.parquet"
    features.to_parquet(processed_file, engine="pyarrow", index=False)
    logger.info(f"Saved processed features to {processed_file}")

    # Use clean data (not aggregated features) for time series forecasting
    if video_id:
        df_video = clean[clean["video_id"] == video_id].copy()
    else:
        # Pick a video with enough data points for Prophet
        video_counts = clean["video_id"].value_counts()
        video_id = video_counts[video_counts >= 2].index[0]
        df_video = clean[clean["video_id"] == video_id].copy()

    ts_df = df_video[["trending_date", "views"]].rename(
        columns={"trending_date": "ds", "views": "y"}
    )

    prophet_model = ProphetForecast()
    prophet_model.fit(ts_df, target_col="y")

    forecast = prophet_model.predict(periods=forecast_days)

    metrics = prophet_model.evaluate(ts_df, target_col="y")
    logger.info(f"Historical fit metrics: {metrics}")

    forecast_file = f"data/processed/{video_id or 'sample'}_forecast.csv"
    forecast.to_csv(forecast_file, index=False)
    logger.info(f"Saved forecast to {forecast_file}")

    print(f"Forecast for next {forecast_days} days for {video_id or 'sample'}:")
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_days))
    print("Historical fit metrics:", metrics)

if __name__ == "__main__":
    main()
