import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from california_houseprice_prediction.domain.train_linear_regression_model import train_and_log_model

def test_train_and_log_base_model():
    # Charger des données factices pour le test
    data = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    # Tester que la fonction s'exécute sans erreur
    train_and_log_model(X_train, X_test, y_train, y_test)