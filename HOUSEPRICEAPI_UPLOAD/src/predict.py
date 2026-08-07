"""
Prediction module.
Loads the trained model and makes predictions.
"""

import joblib
import numpy as np

from src.config import MODEL_PATH

# Load model once when the application starts
model = joblib.load(MODEL_PATH)


def predict_house_price(features: list):

    """
    Predict house price using the trained model.

    Parameters
    ----------
    features : list
        List containing all house features.

    Returns
    -------
    float
        Predicted house price.
    """

    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)

    return float(prediction[0])