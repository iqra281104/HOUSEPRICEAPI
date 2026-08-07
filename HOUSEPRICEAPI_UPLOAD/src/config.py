from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = BASE_DIR / "data" / "Housing.csv"

# Folder where trained models will be saved
MODEL_DIR = BASE_DIR / "models"

# Trained model file
MODEL_PATH = MODEL_DIR / "model.pkl"

# MLflow experiment name
MLFLOW_EXPERIMENT_NAME = "House Price Prediction"

# Random seed for reproducibility
RANDOM_STATE = 42

# Test data percentage
TEST_SIZE = 0.20