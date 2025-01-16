import mlflow.pyfunc
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
#from california_houseprice_prediction.infrastructure import load_and_split_data

def fetch_model(model_name, alias, X_test, y_test):
    """
    Récupère un modèle depuis le MLflow Model Registry et évalue ses performances.

    Args:
        model_name (str): Nom du modèle dans le Registry.
        alias (str): Alias du modèle (par exemple, "champion").
        X_test: Données d'entrée de test (DataFrame ou tableau NumPy).
        y_test: Valeurs cibles de test (DataFrame ou tableau NumPy).

    Returns:
        Tuple: Modèle chargé et métriques de performance (RMSE, MAE).
    """
    # Forcer X_test et y_test à être des DataFrames ou des tableaux NumPy
    if not isinstance(X_test, (pd.DataFrame, np.ndarray)):
        raise TypeError("X_test doit être un DataFrame ou un tableau NumPy.")
    if not isinstance(y_test, (pd.Series, np.ndarray)):
        raise TypeError("y_test doit être une Series ou un tableau NumPy.")

    # Convertir y_test en tableau NumPy si c'est une Series
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    # Récupérer le modèle depuis le Registry
    model_uri = f"models:/{model_name}@{alias}"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Modèle chargé : {model_name} (alias: {alias})")

    # Faire des prédictions
    y_pred = model.predict(X_test)

    # Calculer les métriques
    metrics = {
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
    }

    # Afficher les métriques
    print(f"RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")
    return model, metrics