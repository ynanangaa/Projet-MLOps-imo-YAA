import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from split_train_test import X_train, X_test, y_train, y_test

# Suivi de l'expérience
with mlflow.start_run():
    reg = LinearRegression().fit(X_train, y_train)

    # Enregistrer les métriques et hyperparamètres
    r_squared = reg.score(X_train, y_train)
    y_pred = reg.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    #mlflow.log_param(", n_estimators)
    #mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("R²",r_squared)
    mlflow.log_metric("RMSE",rmse)
    mlflow.log_metric("MAE",mae)

    # Enregistrer le modèle
    mlflow.sklearn.log_model(reg, "model")