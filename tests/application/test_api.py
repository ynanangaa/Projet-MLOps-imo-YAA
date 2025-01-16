import pytest
from fastapi.testclient import TestClient
from california_houseprice_prediction.application import (
    app,
    CaliforniaHousingInput,
)
import numpy as np

# Client de test pour FastAPI
client = TestClient(app)


# Test pour la route de prédiction avec des données valides
def test_predict_valid_input():
    # Données d'entrée valides
    input_data = {
        "median_income": 8.3252,
        "house_age": 41.0,
        "avg_rooms": 6.984127,
        "avg_bedrooms": 1.023810,
        "population": 322.0,
        "avg_occupancy": 2.555556,
        "latitude": 37.88,
        "longitude": -122.23,
    }

    # Envoyer une requête POST à l'endpoint de prédiction
    response = client.post("/predict", json=input_data)

    # Vérifier que la réponse est réussie (status code 200)
    assert response.status_code == 200

    # Vérifier que la réponse contient une prédiction
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)


# Test pour une requête avec des données invalides
def test_predict_invalid_input():
    # Données d'entrée invalides (par exemple, un champ manquant)
    invalid_input = {
        "median_income": 8.3252,
        "house_age": 41.0,
        "avg_rooms": 6.984127,
        "avg_bedrooms": 1.023810,
        "population": 322.0,
        "avg_occupancy": 2.555556,
        # "latitude" manquant
        "longitude": -122.23,
    }

    # Envoyer une requête POST avec des données invalides
    response = client.post("/predict", json=invalid_input)

    # Vérifier que la réponse est une erreur (status code 422)
    assert response.status_code == 422


# Test pour une requête avec des types de données incorrects
def test_predict_invalid_types():
    # Données d'entrée avec des types incorrects
    invalid_input = {
        "median_income": "invalid",  # Doit être un float
        "house_age": 41.0,
        "avg_rooms": 6.984127,
        "avg_bedrooms": 1.023810,
        "population": 322.0,
        "avg_occupancy": 2.555556,
        "latitude": 37.88,
        "longitude": -122.23,
    }

    # Envoyer une requête POST avec des types incorrects
    response = client.post("/predict", json=invalid_input)

    # Vérifier que la réponse est une erreur (status code 422)
    assert response.status_code == 422
