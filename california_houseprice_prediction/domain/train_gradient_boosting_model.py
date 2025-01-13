import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from california_houseprice_prediction.infrastructure.split_data_train_test import X_train, X_test, y_train, y_test

# Définir les hyperparamètres
n_estimators = 100
max_depth = 5
learning_rate = 0.3

# Suivi de l'expérience
mlflow.set_experiment("gradient_boosting_model")
with mlflow.start_run():
    gb_reg = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
    gb_reg.fit(X_train, y_train)

    # Enregistrer les métriques et hyperparamètres
    y_train_pred = gb_reg.predict(X_train)
    r_squared = r2_score(y_train, y_train_pred)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("learning_rate", learning_rate)
    y_pred = gb_reg.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mlflow.log_metric("R²",r_squared)
    mlflow.log_metric("RMSE",rmse)
    mlflow.log_metric("MAE",mae)

    # Enregistrer le modèle
    mlflow.sklearn.log_model(gb_reg, "model")
