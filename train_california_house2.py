import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from split_train_test import X_train, X_test, y_train, y_test

# Définir les hyperparamètres
n_estimators = 100
max_depth = 5

# Suivi de l'expérience
with mlflow.start_run():
    rf_reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_reg.fit(X_train, y_train)

    # Enregistrer les métriques et hyperparamètres
	y_train_pred = rf_reg.predict(X_train)
	r_squared = r2_score(y_train, y_train_pred)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    y_pred = rf_reg.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mlflow.log_metric("R²",r_squared)
    mlflow.log_metric("RMSE",rmse)
    mlflow.log_metric("MAE",mae)

    # Enregistrer le modèle
    mlflow.sklearn.log_model(rf_reg, "model")
