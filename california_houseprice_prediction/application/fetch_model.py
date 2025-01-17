import mlflow.pyfunc
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from california_houseprice_prediction.domain import (
    REGISTERED_MODEL_NAME,
    REGISTERED_MODEL_ALIAS,
)


def fetch_model(
    model_name=REGISTERED_MODEL_NAME,
    alias=REGISTERED_MODEL_ALIAS,
    X_test=None,
    y_test=None,
):
    """
    Récupère un modèle depuis le MLflow Model Registry et évalue ses performances.

    Args:
        model_name (str): Nom du modèle dans le Registry.
        alias (str): Alias du modèle (par exemple, "champion").
        X_test: Données d'entrée de test (DataFrame ou tableau NumPy). Si None, les métriques ne sont pas calculées.
        y_test: Valeurs cibles de test (DataFrame ou tableau NumPy). Si None, les métriques ne sont pas calculées.

    Returns:
        Tuple: Modèle chargé et métriques de performance (RMSE, MAE). Si X_test ou y_test est None, metrics vaut None.
    """
    # Si X_test ou y_test est None, retourner None pour les métriques
    if X_test is None or y_test is None:
        model_uri = f"models:/{model_name}@{alias}"
        model = mlflow.pyfunc.load_model(model_uri)
        return model, None

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

    # Faire des prédictions
    y_pred = model.predict(X_test)

    # Calculer les métriques
    metrics = {
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
    }

    return model, metrics


if __name__ == "__main__":
    # Charger les données de test (à implémenter ou importer depuis un module)
    # X_test, y_test = load_and_split_data()  # Exemple d'importation de données

    # Données de test factices pour l'exemple (à remplacer par des données réelles)
    X_test = np.random.rand(100, 8)  # 100 échantillons, 8 caractéristiques
    y_test = np.random.rand(100)  # 100 valeurs cibles

    # Utiliser les variables importées pour le nom du modèle et l'alias
    model_name = REGISTERED_MODEL_NAME
    alias = REGISTERED_MODEL_ALIAS

    # Charger et évaluer le modèle
    model, metrics = fetch_model(model_name, alias, X_test, y_test)

    # Afficher les résultats
    print(f"Modèle évalué : {model_name} (alias: {alias})")
    print(
        f"Métriques : [ RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f} ]"
    )
