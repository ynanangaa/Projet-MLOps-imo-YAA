import pytest
from fastapi.testclient import TestClient
from california_houseprice_prediction.application.main import app, CaliforniaHousingInput  # Importer depuis le module

# Initialiser le client de test
client = TestClient(app)

# Données de test valides
valid_input_data = {
    "median_income": 8.3252,
    "house_age": 41.0,
    "avg_rooms": 6.984127,
    "avg_bedrooms": 1.023810,
    "population": 322.0,
    "avg_occupancy": 2.555556,
    "latitude": 37.88,
    "longitude": -122.23,
}

# Données de test invalides (manque un champ)
invalid_input_data = {
    "median_income": 8.3252,
    "house_age": 41.0,
    "avg_rooms": 6.984127,
    "avg_bedrooms": 1.023810,
    "population": 322.0,
    "avg_occupancy": 2.555556,
    # "latitude" manquant
    "longitude": -122.23,
}

# Test pour une prédiction réussie
def test_predict_success():
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200  # Vérifier que la requête a réussi
    assert "prediction" in response.json()  # Vérifier que la réponse contient une prédiction
    assert isinstance(response.json()["prediction"], float)  # Vérifier que la prédiction est un float

# Test pour une entrée invalide
def test_predict_invalid_input():
    response = client.post("/predict", json=invalid_input_data)
    assert response.status_code == 422  # Vérifier que la requête a échoué (validation des données)
    assert "detail" in response.json()  # Vérifier que la réponse contient un message d'erreur

# Test pour une entrée vide
def test_predict_empty_input():
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Vérifier que la requête a échoué (validation des données)
    assert "detail" in response.json()  # Vérifier que la réponse contient un message d'erreur