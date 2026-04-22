from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os
import time

# Charger le modèle au démarrage
while not os.path.exists("models/model.pkl"):
    print("Waiting for model...")
    time.sleep(1)
    
model = joblib.load("models/model.pkl")

app = FastAPI()


class CaliforniaHousingInput(BaseModel):
    median_income: float
    house_age: float
    avg_rooms: float
    avg_bedrooms: float
    population: float
    avg_occupancy: float
    latitude: float
    longitude: float


@app.get("/")
def root():
    return {"message": "House Price Prediction API is running"}


@app.post("/predict")
def predict(input_data: CaliforniaHousingInput):
    features = np.array([[
        input_data.median_income,
        input_data.house_age,
        input_data.avg_rooms,
        input_data.avg_bedrooms,
        input_data.population,
        input_data.avg_occupancy,
        input_data.latitude,
        input_data.longitude,
    ]])

    prediction = model.predict(features)[0] * 100000

    return {"prediction": prediction}