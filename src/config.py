from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Legacy single-country paths (for backward compatibility)
RAW_FILE = DATA_RAW / "DE_youtube_trending_data.csv"
PROCESSED_FILE = DATA_PROCESSED / "germany_features.parquet"

# Feature engineering
EARLY_DAYS = 3  # Days used for early engagement features

# Model parameters
RANDOM_STATE = 42
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 12
RF_CV_FOLDS = 5
PROPHET_FORECAST_DAYS = 7

# YouTube category mapping (standard YouTube API categories)
CATEGORY_NAMES = {
    1: "Film & Animation",
    2: "Autos & Vehicles",
    10: "Music",
    15: "Pets & Animals",
    17: "Sports",
    19: "Travel & Events",
    20: "Gaming",
    22: "People & Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News & Politics",
    26: "Howto & Style",
    27: "Education",
    28: "Science & Technology",
    29: "Nonprofits & Activism",
}

# Data validation thresholds
MIN_VIEWS_THRESHOLD = 0
MAX_VIEWS_THRESHOLD = 2_000_000_000  # 2 billion (reasonable YouTube max)
MIN_TREND_DAYS_FOR_PROPHET = 2  # Minimum data points for time series