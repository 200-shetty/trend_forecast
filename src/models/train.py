"""
RandomForest model training for YouTube trend duration prediction.

Addresses RQ1: Can early engagement metrics predict total trending duration?
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    GridSearchCV
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

from src.config import (
    RANDOM_STATE,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_CV_FOLDS,
    MODELS_DIR
)

logger = logging.getLogger(__name__)


class RandomForestTrendPredictor:
    """
    RandomForest model for predicting video trend duration.

    Predicts trend_days (how many days a video will stay in trending)
    based on early engagement features from the first 3 days.
    """

    def __init__(
        self,
        n_estimators: int = RF_N_ESTIMATORS,
        max_depth: int = RF_MAX_DEPTH,
        random_state: int = RANDOM_STATE
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names = None
        self.is_fitted = False
        self.cv_results = None

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "trend_days",
        use_cv: bool = True
    ) -> Dict:
        """
        Train the model with optional cross-validation.

        Args:
            df: Feature DataFrame (must include target column)
            target_col: Name of target column
            use_cv: Whether to perform cross-validation

        Returns:
            Dictionary with training metrics
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]

        self.feature_names = list(X.columns)

        if use_cv:
            self.cv_results = self._cross_validate(X, y)
            logger.info(f"CV Results - MAE: {self.cv_results['cv_mae_mean']:.3f} "
                       f"(±{self.cv_results['cv_mae_std']:.3f})")

        # Train on full data
        self.model.fit(X, y)
        self.is_fitted = True

        # Training metrics
        train_preds = self.model.predict(X)
        metrics = {
            "train_mae": mean_absolute_error(y, train_preds),
            "train_rmse": np.sqrt(mean_squared_error(y, train_preds)),
            "train_r2": r2_score(y, train_preds),
            "n_samples": len(df),
            "n_features": len(self.feature_names)
        }

        if self.cv_results:
            metrics.update(self.cv_results)

        logger.info(f"Model trained on {len(df):,} samples with {len(self.feature_names)} features")

        return metrics

    def _cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Perform k-fold cross-validation."""
        kfold = KFold(n_splits=RF_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        # Multiple scoring metrics
        mae_scores = -cross_val_score(
            self.model, X, y, cv=kfold, scoring="neg_mean_absolute_error"
        )
        r2_scores = cross_val_score(self.model, X, y, cv=kfold, scoring="r2")

        return {
            "cv_mae_mean": mae_scores.mean(),
            "cv_mae_std": mae_scores.std(),
            "cv_r2_mean": r2_scores.mean(),
            "cv_r2_std": r2_scores.std(),
            "cv_folds": RF_CV_FOLDS
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances ranked by importance.

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        importance["rank"] = range(1, len(importance) + 1)
        importance["cumulative_importance"] = importance["importance"].cumsum()

        return importance

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """
        Evaluate model on test data.

        Returns comprehensive metrics including error distribution.
        """
        preds = self.predict(X_test)

        errors = y_test - preds

        metrics = {
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds),
            "MAPE": mean_absolute_percentage_error(y_test, preds) * 100,
            "median_error": np.median(np.abs(errors)),
            "error_std": errors.std(),
            "n_test_samples": len(y_test)
        }

        return metrics

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Save model to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        filepath = filepath or (MODELS_DIR / "random_forest_model.joblib")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "cv_results": self.cv_results
        }, filepath)

        logger.info(f"Model saved to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "RandomForestTrendPredictor":
        """Load model from disk."""
        data = joblib.load(filepath)

        predictor = cls()
        predictor.model = data["model"]
        predictor.feature_names = data["feature_names"]
        predictor.cv_results = data.get("cv_results")
        predictor.is_fitted = True

        logger.info(f"Model loaded from {filepath}")
        return predictor


def train_model(df: pd.DataFrame) -> Tuple:
    """
    Legacy function for backward compatibility.

    Returns model and test data split.
    """
    X = df.drop(columns=["trend_days"])
    y = df["trend_days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model, X_test, y_test


def tune_hyperparameters(
    df: pd.DataFrame,
    target_col: str = "trend_days",
    param_grid: Optional[Dict] = None
) -> Dict:
    """
    Perform hyperparameter tuning with GridSearchCV.

    Args:
        df: Feature DataFrame
        target_col: Target column name
        param_grid: Custom parameter grid (optional)

    Returns:
        Best parameters and CV results
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [8, 12, 16, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }

    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=RF_CV_FOLDS,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1
    )

    logger.info("Starting hyperparameter tuning...")
    grid_search.fit(X, y)

    results = {
        "best_params": grid_search.best_params_,
        "best_score": -grid_search.best_score_,
        "cv_results": pd.DataFrame(grid_search.cv_results_)
    }

    logger.info(f"Best params: {results['best_params']}")
    logger.info(f"Best MAE: {results['best_score']:.3f}")

    return results
