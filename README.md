# YouTube Trend Forecast

A machine learning pipeline that predicts how long YouTube videos will stay on the trending list and forecasts future view counts using time series analysis.

## Overview

This project analyzes YouTube trending data to:
1. **Predict trend duration** - How many days will a video stay trending? (RandomForest)
2. **Forecast future views** - What will the view count be in the next 7 days? (Prophet)

## Project Structure

```
youtube_trend_forecast/
├── app.py                  # Streamlit web dashboard
├── run_pipeline.py         # CLI entry point for running the full pipeline
├── requirements.txt        # Python dependencies
├── README.md
│
├── data/
│   ├── raw/                # Raw CSV files (e.g., DE_youtube_trending_data.csv)
│   └── processed/          # Output: features parquet, forecast CSVs
│
└── src/
    ├── config.py           # Configuration constants (paths, parameters)
    │
    ├── etl/                # Extract-Transform-Load pipeline
    │   ├── extract.py      # Load raw CSV data by country
    │   ├── transform.py    # Clean data: parse dates, remove nulls/duplicates
    │   └── load.py         # Save processed features to parquet
    │
    ├── features/
    │   └── build_features.py   # Engineer ML features from first 3 trending days
    │
    └── models/
        ├── prophet_model.py    # Time series forecasting with Facebook Prophet
        ├── train.py            # Train RandomForest regression model
        └── evaluate.py         # Compute MAE, RMSE, R² metrics
```

## Data Flow

```
Raw CSV → Extract → Transform → Build Features → Train/Predict
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
              RandomForest                          Prophet
          (predict trend_days)                 (forecast views)
```

## Features Engineered

Features are computed from the **first 3 days** of trending to avoid data leakage:

| Feature | Description |
|---------|-------------|
| `early_views` | Average daily views in first 3 trending days |
| `early_likes` | Average daily likes in first 3 trending days |
| `early_comments` | Average daily comments in first 3 trending days |
| `max_views` | Peak views reached in first 3 days |
| `categoryId` | YouTube category (numeric) |
| `days_since_publish` | Days between upload and first trending |
| `likes_view_ratio` | Engagement: likes / views |
| `comments_view_ratio` | Engagement: comments / views |
| `trend_days` | **Target** - Total days the video stayed trending |

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd youtube_trend_forecast

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. CLI Pipeline

Run the full ETL + forecasting pipeline:

```bash
python run_pipeline.py
```

This will:
- Load and clean the raw data
- Build features and save to `data/processed/`
- Fit a Prophet model on a sample video
- Output a 7-day forecast to `data/processed/<video_id>_forecast.csv`

### 2. Streamlit Dashboard

Launch the interactive web app:

```bash
streamlit run app.py
```

Features:
- View aggregated video statistics with titles and channel names
- See 7-day view forecast with confidence intervals
- Download forecast as CSV

### 3. Train RandomForest Model

```python
from src.etl.extract import extract_raw_data
from src.etl.transform import transform_raw_data
from src.features.build_features import build_features
from src.models.train import train_model
from src.models.evaluate import evaluate

# Load and process data
raw = extract_raw_data("DE")
clean = transform_raw_data(raw)
features = build_features(clean, for_ml=True)  # Numeric features only

# Train and evaluate
model, X_test, y_test = train_model(features)
metrics = evaluate(model, X_test, y_test)
print(metrics)  # {'MAE': 1.08, 'RMSE': 1.41, 'R2': 0.178}
```

## Models

### RandomForest Regressor
- **Purpose**: Predict total trending duration (`trend_days`)
- **Input**: Aggregated features from first 3 days
- **Output**: Predicted number of days video will trend
- **Performance**: R² ≈ 0.18 (virality is inherently hard to predict)

### Facebook Prophet
- **Purpose**: Forecast future view counts
- **Input**: Daily view time series for a specific video
- **Output**: 7-day forecast with confidence intervals
- **Performance**: R² ≈ 0.99 on training data (risk of overfitting)

## Configuration

Edit `src/config.py` to modify:

```python
EARLY_DAYS = 3        # Days of data used for feature engineering
RANDOM_STATE = 42     # Reproducibility seed
DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"
```

## Data

**Note:** Data files are not included in this repository due to size limitations.

### Download the Data

1. Download the YouTube Trending Dataset from [Kaggle](https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset)
2. Place `DE_youtube_trending_data.csv` in `data/raw/`

```bash
mkdir -p data/raw data/processed
# Place your downloaded CSV in data/raw/
```

### Expected Columns

`video_id`, `publishedAt`, `trending_date`, `view_count`, `likes`, `comment_count`, `categoryId`, `title`, `channelTitle`

## Requirements

- Python 3.10+
- pandas
- scikit-learn
- prophet
- streamlit
- pyarrow