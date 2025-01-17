import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error,
)
from california_houseprice_prediction.infrastructure import (
    load_and_split_data,
    EXPERIMENT_NAME,
)
from .log_params_metrics_model import (
    log_parameters,
    log_metrics,
    log_model,
)


def train_and_log_random_forest_model(
    X_train,
    X_test,
    y_train,
    y_test,
    n_estimators=100,
    max_depth=5,
    max_features=None,
):
    """
    Entraîne un modèle de Random Forest et enregistre les métriques et le modèle avec MLflow.

    Args:
        X_train, X_test, y_train, y_test: Données d'entraînement et de test.
        n_estimators (int): Nombre d'estimateurs pour le modèle.
        max_depth (int): Profondeur maximale des arbres.
        max_features (str, int, float or None): Nombre de features à considérer pour le split.
    """

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        # Initialiser et entraîner le modèle
        rf_reg = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            random_state=42,
        )
        rf_reg.fit(X_train, y_train)

        # Enregistrer les hyperparamètres
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_features": max_features,
        }
        log_parameters(params)

        # Calculer et enregistrer les métriques
        y_train_pred = rf_reg.predict(X_train)
        y_pred = rf_reg.predict(X_test)
        metrics = {
            "R2": r2_score(y_train, y_train_pred),
            "RMSE": root_mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
        }
        log_metrics(metrics)

        # Enregistrer le modèle
        log_model(rf_reg, "model")
        print("Modèle Random Forest entraîné et enregistré avec succès.")


if __name__ == "__main__":
    # Charger et diviser les données
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Entraîner et logger le modèle
    train_and_log_random_forest_model(X_train, X_test, y_train, y_test)
