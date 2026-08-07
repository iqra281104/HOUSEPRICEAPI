# 🏠 House Price Prediction API

A Machine Learning project that predicts house prices using **Linear Regression**, tracks experiments using **MLflow**, and deploys predictions through **FastAPI**.

---

## Project Overview

This project demonstrates an end-to-end Machine Learning workflow:

- Data preprocessing
- Feature encoding
- Model training
- Model evaluation
- Experiment tracking with MLflow
- API deployment using FastAPI
- API testing with Swagger UI and Postman

---

## Dataset

Dataset: Housing Price Prediction Dataset

Features:

- Area
- Bedrooms
- Bathrooms
- Stories
- Main Road
- Guest Room
- Basement
- Hot Water Heating
- Air Conditioning
- Parking
- Preferred Area
- Furnishing Status

Target:

- House Price

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- MLflow
- Joblib
- Pydantic

---

## Project Structure

HOUSEPRICEAPI/

│

├── app.py

├── train.py

├── requirements.txt

├── README.md

├── data/

│ └── Housing.csv

├── models/

│ └── model.pkl

├── docs/

├── notebooks/

├── src/

│ ├── config.py

│ ├── preprocess.py

│ ├── predict.py

│ ├── evaluate.py

│ ├── utils.py

│ └── **init**.py

└── tests/

---

## Model

Algorithm Used:

Linear Regression

---

## Model Performance

RMSE:

≈ 1,331,071

MAE:

≈ (Your MAE)

R² Score:

≈ 0.65

---

## MLflow

The project uses MLflow to track:

- Parameters
- Metrics
- Model artifacts

Start MLflow:

```bash
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

---

## FastAPI

Run:

```bash
uvicorn app:app --reload
```

Open Swagger:

```
http://127.0.0.1:8000/docs
```

---

## Example Request

```json
{
  "area":7420,
  "bedrooms":4,
  "bathrooms":2,
  "stories":3,
  "mainroad":1,
  "guestroom":0,
  "basement":0,
  "hotwaterheating":0,
  "airconditioning":1,
  "parking":2,
  "prefarea":1,
  "furnishingstatus":2
}
```

Example Response

```json
{
  "success": true,
  "predicted_price": 12345678.45,
  "currency": "PKR"
}
```

---

## Future Improvements

- Hyperparameter tuning
- Cross-validation
- One-Hot Encoding
- Docker deployment
- Cloud deployment

---

## Author

BS Mathematics Machine Learning Assignment