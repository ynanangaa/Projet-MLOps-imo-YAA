import mlflow
from sklearn.ensemble import GradientBoostingRegressor
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


def train_and_log_gradient_boosting_model(
    X_train,
    X_test,
    y_train,
    y_test,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.3,
    max_features=None,
):
    """
    Entraîne un modèle de Gradient Boosting et enregistre les métriques et le modèle avec MLflow.

    Args:
        X_train, X_test, y_train, y_test: Données d'entraînement et de test.
        n_estimators (int): Nombre d'estimateurs pour le modèle.
        max_depth (int): Profondeur maximale des arbres.
        learning_rate (float): Taux d'apprentissage.
    """

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        # Initialiser et entraîner le modèle
        gb_reg = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            learning_rate=learning_rate,
            random_state=42,
        )
        gb_reg.fit(X_train, y_train)

        # Enregistrer les hyperparamètres
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "max_features": max_features,
        }
        log_parameters(params)

        # Calculer et enregistrer les métriques
        y_train_pred = gb_reg.predict(X_train)
        y_pred = gb_reg.predict(X_test)
        metrics = {
            "R2": r2_score(y_train, y_train_pred),
            "RMSE": root_mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
        }
        log_metrics(metrics)

        # Enregistrer le modèle
        log_model(gb_reg, "model")
        print("Modèle Gradient Boosting entraîné et enregistré avec succès.")


if __name__ == "__main__":
    # Charger et diviser les données
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Entraîner et logger le modèle
    train_and_log_gradient_boosting_model(X_train, X_test, y_train, y_test)
