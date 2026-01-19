from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

RAW_FILE = DATA_RAW / "DE_youtube_trending_data.csv"
PROCESSED_FILE = DATA_PROCESSED / "germany_features.parquet"

EARLY_DAYS = 3
RANDOM_STATE = 42