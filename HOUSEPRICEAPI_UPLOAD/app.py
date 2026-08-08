from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.predict import predict_house_price
import time
from monitoring.metrics import record_request, get_metrics
app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices using a trained Linear Regression model.",
    version="1.0.0"
)
@app.middleware("http")
async def monitor_requests(request: Request, call_next):

    start_time = time.time()

    response = await call_next(request)

    response_time = time.time() - start_time

    record_request(
        request.url.path,
        response_time
    )

    return response

class HouseFeatures(BaseModel):

    area: int = Field(..., gt=0, description="Area of the house")
    bedrooms: int = Field(..., ge=1)
    bathrooms: int = Field(..., ge=1)
    stories: int = Field(..., ge=1)

    mainroad: int = Field(..., ge=0, le=1)
    guestroom: int = Field(..., ge=0, le=1)
    basement: int = Field(..., ge=0, le=1)
    hotwaterheating: int = Field(..., ge=0, le=1)
    airconditioning: int = Field(..., ge=0, le=1)

    parking: int = Field(..., ge=0, le=5)

    prefarea: int = Field(..., ge=0, le=1)

    furnishingstatus: int = Field(..., ge=0, le=2)


@app.get("/")
def home():

    return {
        "message": "House Price Prediction API is running successfully."
    }


@app.get("/health")
def health():

    return {
        "status": "Healthy",
        "model": "Linear Regression",
        "api": "Running"
    }


@app.post("/predict")
def predict(data: HouseFeatures):

    try:

        features = [
            data.area,
            data.bedrooms,
            data.bathrooms,
            data.stories,
            data.mainroad,
            data.guestroom,
            data.basement,
            data.hotwaterheating,
            data.airconditioning,
            data.parking,
            data.prefarea,
            data.furnishingstatus,
        ]

        prediction = predict_house_price(features)

        return {
            "success": True,
            "predicted_price": round(prediction, 2),
            "currency": "PKR"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
@app.get("/metrics")
def metrics():

    return get_metrics()        