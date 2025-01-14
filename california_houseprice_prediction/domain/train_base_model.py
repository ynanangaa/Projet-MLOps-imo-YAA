import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import california_houseprice_prediction.infrastructure as infrastructure
from infrastructure.split_data_train_test import load_and_split_data


def train_and_log_base_model(X_train, X_test, y_train, y_test):
    """
    Entraîne un modèle de régression linéaire et enregistre les métriques et \
        le modèle avec MLflow.

    Args:
        X_train, X_test, y_train, y_test: Données d'entraînement et de test.
    """
    mlflow.set_experiment("base_model")
    with mlflow.start_run():
        reg = LinearRegression().fit(X_train, y_train)

        # Enregistrer les métriques et hyperparamètres
        r_squared = reg.score(X_train, y_train)
        y_pred = reg.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        mlflow.log_metric("R²", r_squared)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)

        # Enregistrer le modèle
        mlflow.sklearn.log_model(reg, "model")


if __name__ == "__main__":
    # Charger et diviser les données
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Entraîner et logger le modèle
    train_and_log_base_model(X_train, X_test, y_train, y_test)
