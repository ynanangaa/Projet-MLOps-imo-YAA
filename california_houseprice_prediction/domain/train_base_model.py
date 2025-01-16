import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from california_houseprice_prediction.infrastructure import load_and_split_data
from california_houseprice_prediction.domain.log_params_metrics_model import (
    log_parameters, log_metrics, log_model
)

def train_and_log_base_model(X_train, X_test, y_train, y_test):
    """
    Entraîne un modèle de régression linéaire et enregistre les métriques et le modèle avec MLflow.

    Args:
        X_train, X_test, y_train, y_test: Données d'entraînement et de test.
    """

    mlflow.set_experiment("california-housing")
    with mlflow.start_run():
        # Initialiser et entraîner le modèle
        reg = LinearRegression()
        reg.fit(X_train, y_train)

        # Enregistrer les hyperparamètres (ici, il n'y en a pas pour LinearRegression)
        params = {}  # LinearRegression n'a pas d'hyperparamètres significatifs à logger
        log_parameters(params)

        # Calculer et enregistrer les métriques
        y_train_pred = reg.predict(X_train)
        y_pred = reg.predict(X_test)
        metrics = {
            "R2": r2_score(y_train, y_train_pred),
            "RMSE": root_mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred)
        }
        log_metrics(metrics)

        # Enregistrer le modèle
        log_model(reg, "base-model")
        print("Modèle de base (régression linéaire) entraîné et enregistré avec succès.")

if __name__ == "__main__":
    # Charger et diviser les données
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Entraîner et logger le modèle
    train_and_log_base_model(X_train, X_test, y_train, y_test)