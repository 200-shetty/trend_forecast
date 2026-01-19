import streamlit as st
import pandas as pd
from pathlib import Path
from src.models.prophet_model import ProphetForecast
from src.etl.extract import extract_raw_data
from src.etl.transform import transform_raw_data
from src.features.build_features import build_features
from src.config import DATA_PROCESSED, RAW_FILE

st.set_page_config(page_title="YouTube Trend Forecast", layout="wide")
st.title("YouTube Trend Forecast Dashboard")
st.caption("Analyzing German YouTube Trending Data")

# Load / Cache data
@st.cache_data
def load_data():
    raw_df = extract_raw_data("DE")
    clean_df = transform_raw_data(raw_df)
    features_df = build_features(clean_df)
    # Keep video metadata for display
    video_info = raw_df[["video_id", "title", "channelTitle"]].drop_duplicates("video_id")
    return clean_df, features_df, video_info

clean_df, features_df, video_info = load_data()

# Video Statistics Summary
st.subheader("Video Statistics Summary")
st.markdown("""
This table shows **aggregated features** for each YouTube video based on their **first 3 days** of trending.
These features are used to predict how long a video will stay on the trending list.

| Column | Description |
|--------|-------------|
| `title` | Video title |
| `channelTitle` | YouTube channel name |
| `early_views` | Average daily views in first 3 trending days |
| `early_likes` | Average daily likes in first 3 trending days |
| `max_views` | Peak views reached in first 3 days |
| `trend_days` | **Target variable** - Total days the video stayed trending |
| `days_since_publish` | Days between video upload and first trending appearance |
""")

# Merge features with video titles for display
display_df = features_df.merge(video_info, on="video_id", how="left")
display_cols = ["title", "channelTitle", "early_views", "early_likes", "max_views", "trend_days", "days_since_publish"]
st.dataframe(display_df[display_cols].head(10))

# Pick a video with the most data points for Prophet
video_counts = clean_df["video_id"].value_counts()
video_id = video_counts[video_counts >= 2].index[0]
df_video = clean_df[clean_df["video_id"] == video_id].copy()

# Get video title and channel for display
video_title = video_info[video_info["video_id"] == video_id]["title"].values[0]
channel_name = video_info[video_info["video_id"] == video_id]["channelTitle"].values[0]

st.sidebar.success(f"**Forecasting:**\n\n{video_title}\n\nby *{channel_name}*\n\n({len(df_video)} data points)")

# Prepare for Prophet - forecast views over time
ts_df = df_video[["trending_date", "views"]].rename(
    columns={"trending_date": "ds", "views": "y"}
)

# Fit Prophet
prophet_model = ProphetForecast()
prophet_model.fit(ts_df, target_col="y")
forecast = prophet_model.predict(periods=7)

st.subheader("Views Forecast (Next 7 days)")
st.markdown("""
This chart shows the **predicted view count** for the selected video over the next 7 days using Facebook Prophet time series forecasting.

| Line | Description |
|------|-------------|
| **yhat** | Predicted view count (main forecast line) |
| **yhat_upper** | Upper bound of 80% confidence interval |
| **yhat_lower** | Lower bound of 80% confidence interval |

The shaded area between upper and lower bounds represents uncertainty — the wider the gap, the less confident the model is about that prediction.
""")
st.line_chart(forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])

# Evaluate on historical
metrics = prophet_model.evaluate(ts_df, target_col="y")
st.subheader("Model Performance Metrics")
st.markdown("""
These metrics show how well the model fits the **historical data** (training set, not future predictions).

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | Mean Absolute Error | Average prediction error in views |
| **RMSE** | Root Mean Squared Error | Penalizes larger errors more heavily |
| **R²** |  Coefficient of Determination | How much variance the model explains (1.0 = perfect) |

""")
st.write(metrics)

# Download predictions
st.download_button(
    label="Download Forecast CSV",
    data=forecast.to_csv(index=False),
    file_name="forecast.csv",
    mime="text/csv"
)
