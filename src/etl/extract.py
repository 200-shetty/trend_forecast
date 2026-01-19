import pandas as pd
from pathlib import Path
from src.config import RAW_FILE

def extract_raw_data(country="DE"):
    file_path = Path(f"data/raw/{country}_youtube_trending_data.csv")
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file for {country} not found: {file_path}")
    
    df = pd.read_csv(file_path)
    return df
