from california_houseprice_prediction.infrastructure import load_and_split_data
from mlflow.models import validate_serving_input, convert_input_example_to_serving_input
import pandas as pd

# Charger les données
X_train, X_test, y_train, y_test = load_and_split_data()

# URI du modèle MLflow
model_uri = 'runs:/f5b6973bebb14e64a668b69359a7fa74/model'

# Extraire un exemple d'entrée (par exemple, la première ligne de X_test)
INPUT_EXAMPLE = X_test.iloc[0:1]  # Prendre la première ligne sous forme de DataFrame

# Convertir l'exemple d'entrée en format de service
serving_payload = convert_input_example_to_serving_input(INPUT_EXAMPLE)

# Valider que le payload fonctionne avec le modèle
validate_serving_input(model_uri, serving_payload)

print("Validation réussie !")