from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np
from .fetch_model import fetch_model

model, _ = fetch_model()

# Initialiser l'application FastAPI
app = FastAPI()


# Schéma pour les données d'entrée
class CaliforniaHousingInput(BaseModel):
    median_income: float
    house_age: float
    avg_rooms: float
    avg_bedrooms: float
    population: float
    avg_occupancy: float
    latitude: float
    longitude: float


# Point de terminaison pour les prédictions
@app.post("/predict")
def predict(input_data: CaliforniaHousingInput):
    # Transformer les données en tableau numpy
    features = np.array(
        [
            [
                input_data.median_income,
                input_data.house_age,
                input_data.avg_rooms,
                input_data.avg_bedrooms,
                input_data.population,
                input_data.avg_occupancy,
                input_data.latitude,
                input_data.longitude,
            ]
        ]
    )
    # Faire une prédiction
    prediction = model.predict(features)
    return {"prediction": prediction[0]}
