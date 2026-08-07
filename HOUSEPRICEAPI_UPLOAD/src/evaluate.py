"""
evaluate.py

Contains helper functions to evaluate regression models.
"""

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import numpy as np


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained regression model.

    Parameters
    ----------
    model : trained sklearn model
    X_test : Test features
    y_test : Actual target values

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }