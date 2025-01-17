import requests


def mlflow_model_serving(
    url="http://localhost:5001/invocations",
    data=None,
    headers=None,
):
    """
    Envoie une requête à l'API de serving MLflow pour obtenir une prédiction.

    Args:
        url (str): URL de l'API MLflow (par défaut : "http://localhost:5001/invocations").
        data (dict): Données d'entrée pour la prédiction (par défaut : données factices).
        headers (dict): En-têtes de la requête (par défaut : {"Content-Type": "application/json"}).

    Returns:
        dict: Réponse de l'API sous forme de dictionnaire.
    """
    # Données par défaut si non fournies
    if data is None:
        data = {
            "inputs": [
                [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]
            ]
        }

    # En-têtes par défaut si non fournis
    if headers is None:
        headers = {"Content-Type": "application/json"}

    # Envoyer la requête POST
    response = requests.post(url, json=data, headers=headers)

    # Vérifier la réponse
    if response.status_code == 200:
        print("Test réussi ! Prédiction :", response.json())
    else:
        print(f"Échec du test. Code de statut : {response.status_code}")
        print("Réponse :", response.text)

    return response.json() if response.status_code == 200 else None


if __name__ == "__main__":
    # Utiliser les valeurs par défaut pour l'exécution principale
    mlflow_model_serving()