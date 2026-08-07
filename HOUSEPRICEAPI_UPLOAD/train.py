"""
train.py

Main training pipeline for the House Price Prediction project.
"""

import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression

from src.preprocess import load_data, encode_data, split_data
from src.evaluate import evaluate_model
from src.config import MODEL_PATH, MLFLOW_EXPERIMENT_NAME


def train_models():

    print("=" * 60)
    print("HOUSE PRICE PREDICTION PROJECT")
    print("=" * 60)

    # Close any previously active MLflow run
    if mlflow.active_run():
        mlflow.end_run()

    # Set MLflow experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # -----------------------------
    # Load and preprocess data
    # -----------------------------
    print("\nLoading dataset...")
    df = load_data()

    print("Encoding dataset...")
    df = encode_data(df)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(df)

    # -----------------------------
    # Create model
    # -----------------------------
    model = LinearRegression()

    print("\nTraining Linear Regression...")

    # -----------------------------
    # MLflow Run
    # -----------------------------
    with mlflow.start_run(run_name="Linear Regression"):

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Log parameters
        mlflow.log_param("model", "Linear Regression")

        # Log metrics
        mlflow.log_metric("RMSE", metrics["RMSE"])
        mlflow.log_metric("MAE", metrics["MAE"])
        mlflow.log_metric("R2", metrics["R2"])

        # Log trained model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model"
        )

    # -----------------------------
    # Save model locally
    # -----------------------------
    joblib.dump(model, MODEL_PATH)

    # -----------------------------
    # Print Results
    # -----------------------------
    print("\n")
    print("=" * 40)
    print("MODEL EVALUATION")
    print("=" * 40)

    print(f"RMSE : {metrics['RMSE']:.2f}")
    print(f"MAE  : {metrics['MAE']:.2f}")
    print(f"R²   : {metrics['R2']:.4f}")

    print("\nModel saved successfully!")
    print(f"Location : {MODEL_PATH}")


if __name__ == "__main__":
    train_models()