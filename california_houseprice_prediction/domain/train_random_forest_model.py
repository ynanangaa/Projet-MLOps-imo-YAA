import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from california_houseprice_prediction.infrastructure import load_and_split_data


def train_and_log_random_forest_model(
    X_train, X_test, y_train, y_test, n_estimators=100, max_depth=5,
    max_features=None
):
    """
    Entraîne un modèle de Random Forest et enregistre les métriques et le \
        modèle avec MLflow.

    Args:
        X_train, X_test, y_train, y_test: Données d'entraînement et de test.
        n_estimators (int): Nombre d'estimateurs pour le modèle.
        max_depth (int): Profondeur maximale des arbres.
    """
    mlflow.set_experiment("random_forest_model")
    with mlflow.start_run():
        # Initialiser et entraîner le modèle
        rf_reg = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42,
            max_features=max_features
        )
        rf_reg.fit(X_train, y_train)

        # Enregistrer les hyperparamètres
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("max_features", max_features)

        # Calculer et enregistrer les métriques
        y_train_pred = rf_reg.predict(X_train)
        r_squared = r2_score(y_train, y_train_pred)
        y_pred = rf_reg.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        mlflow.log_metric("R²", r_squared)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)

        # Enregistrer le modèle
        mlflow.sklearn.log_model(rf_reg, "model")
        print("Modèle Random Forest entraîné et enregistré avec succès.")


if __name__ == "__main__":
    # Charger et diviser les données
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Entraîner et logger le modèle
    train_and_log_random_forest_model(X_train, X_test, y_train, y_test)
