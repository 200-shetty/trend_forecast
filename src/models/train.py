import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE

def train_model(df: pd.DataFrame):
    X = df.drop(columns=["trend_days"])
    y = df["trend_days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model, X_test, y_test
