import pandas as pd
from src.config import PROCESSED_FILE

def load_features(df: pd.DataFrame) -> None:
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
    PROCESSED_FILE,
    engine="pyarrow",
    index=False
)

