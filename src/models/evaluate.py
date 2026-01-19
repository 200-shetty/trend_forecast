from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": rmse,
        "R2": r2_score(y_test, preds),
    }
