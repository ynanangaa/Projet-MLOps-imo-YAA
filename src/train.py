import joblib
import logging
import mlflow
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

def train():
    # Charger les données
    data = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # Hyperparamètres issus du notebook experiments.ipynb
    params = {
        "n_estimators": 120,
        "max_depth": 5,
        "learning_rate": 0.2,
        "max_features": None,
    }

    mlflow.set_experiment("experiment_gradient_boosting")

    with mlflow.start_run():

        # Initialisation modèle
        model = GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            max_features=params["max_features"],
            random_state=42,
        )

        # Entraînement
        model.fit(X_train, y_train)

        # Prédictions
        y_pred = model.predict(X_test)

        # Métriques
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Logging MLflow
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, "model")

        logging.info("✅ Modèle entraîné et loggé avec MLflow")
        logging.info(f"RMSE: {rmse:.4f}")
        logging.info(f"MAE: {mae:.4f}")
        logging.info(f"R2: {r2:.4f}")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")


if __name__ == "__main__":
    train()