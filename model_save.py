import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Charger les données
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

model = mlflow.sklearn.load_model("runs:/a73dd653c0bc42b98f0abcc0b69a08d4/model")
predictions = model.predict(X_test)

print(predictions)



#mlflow models serve -m "models:/<model_name>/Production" --port 1234