import requests


def test_mlflow_serving():
    # URL de l'API MLflow
    url = "http://localhost:5001/invocations"

    # Données d'entrée pour la prédiction
    data = {
        "inputs": [
            [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]
        ]
    }

    # En-têtes de la requête
    headers = {"Content-Type": "application/json"}

    # Envoyer la requête POST
    response = requests.post(url, json=data, headers=headers)

    # Vérifier la réponse
    if response.status_code == 200:
        print("Test réussi ! Prédiction :", response.json())
    else:
        print(f"Échec du test. Code de statut : {response.status_code}")
        print("Réponse :", response.text)


if __name__ == "__main__":
    test_mlflow_serving()
