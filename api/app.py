from fastapi import FastAPI

from api.schema import CustomerData
from api.predict import Predictor

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)

predictor = Predictor()


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is Running"
    }


@app.get("/health")
def health():
    return {
        "status": "Healthy"
    }


@app.post("/predict")
def predict(data: CustomerData):

    prediction = predictor.predict(data.model_dump())

    return {
        "prediction": prediction,
        "churn": "Yes" if prediction == 1 else "No"
    }