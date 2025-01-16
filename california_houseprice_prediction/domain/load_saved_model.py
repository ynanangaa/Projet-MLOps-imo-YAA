import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from california_houseprice_prediction.infrastructure import load_and_split_data

X_train, X_test, y_train, y_test = load_and_split_data()

model = mlflow.sklearn.load_model("runs:/f5b6973bebb14e64a668b69359a7fa74/model")
predictions = model.predict(X_test)

# Calcul des métriques
rmse = root_mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

# Affichage des résultats
print(f"MAE du modèle : {mae}")
print(f"RMSE du modèle : {rmse}")