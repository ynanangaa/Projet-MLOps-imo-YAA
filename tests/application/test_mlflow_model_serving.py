import unittest
from unittest.mock import patch, MagicMock
from california_houseprice_prediction.application.mlflow_model_serving import (
    mlflow_model_serving,
)


class TestMlflowModelServing(unittest.TestCase):

    @patch("requests.post")
    def test_mlflow_model_serving_success(self, mock_post):
        # Simuler une réponse réussie de l'API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predictions": [4.5]
        }  # Exemple de prédiction
        mock_post.return_value = mock_response

        # Appeler la fonction avec des arguments personnalisés
        url = "http://localhost:5001/invocations"
        data = {"inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]}
        headers = {"Content-Type": "application/json"}

        result = mlflow_model_serving(url=url, data=data, headers=headers)

        # Vérifier que la fonction a retourné la réponse simulée
        self.assertEqual(result, {"predictions": [4.5]})

        # Vérifier que la requête a été envoyée correctement
        mock_post.assert_called_once_with(url, json=data, headers=headers)

    @patch("requests.post")
    def test_mlflow_model_serving_failure(self, mock_post):
        # Simuler une réponse d'échec de l'API
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        # Appeler la fonction avec des arguments personnalisés
        url = "http://localhost:5001/invocations"
        data = {"inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]}
        headers = {"Content-Type": "application/json"}

        result = mlflow_model_serving(url=url, data=data, headers=headers)

        # Vérifier que la fonction a retourné None en cas d'échec
        self.assertIsNone(result)

        # Vérifier que la requête a été envoyée correctement
        mock_post.assert_called_once_with(url, json=data, headers=headers)


if __name__ == "__main__":
    unittest.main()
