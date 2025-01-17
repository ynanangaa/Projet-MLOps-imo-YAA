import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from california_houseprice_prediction.application.fetch_model import (
    fetch_model,
)
from california_houseprice_prediction.domain import (
    REGISTERED_MODEL_NAME,
    REGISTERED_MODEL_ALIAS,
)


class TestFetchModel(unittest.TestCase):

    @patch("mlflow.pyfunc.load_model")
    def test_fetch_model(self, mock_load):
        # Simuler le chargement du modèle
        mock_model = MagicMock()
        n_samples = 4128  # Nombre d'échantillons dans tes données réelles
        mock_model.predict.return_value = np.random.rand(
            n_samples
        )  # Prédictions simulées
        mock_load.return_value = mock_model

        # Simuler les données de test
        mock_X_test = np.random.rand(n_samples, 8)  # Tableau NumPy simulé
        mock_y_test = np.random.rand(n_samples)  # Tableau NumPy simulé

        # Appeler la fonction fetch_model
        model, metrics = fetch_model(
            REGISTERED_MODEL_NAME,
            REGISTERED_MODEL_ALIAS,
            mock_X_test,
            mock_y_test,
        )

        # Vérifier que mlflow.pyfunc.load_model a été appelé avec les bons arguments
        mock_load.assert_called_once_with(
            f"models:/{REGISTERED_MODEL_NAME}@{REGISTERED_MODEL_ALIAS}"
        )

        # Vérifier que model.predict a été appelé avec mock_X_test
        mock_model.predict.assert_called_once_with(mock_X_test)

        # Vérifier que les métriques sont retournées
        self.assertIn("RMSE", metrics, "La métrique RMSE n'est pas retournée.")
        self.assertIn("MAE", metrics, "La métrique MAE n'est pas retournée.")

        # Vérifier que le modèle retourné est bien celui simulé
        self.assertEqual(
            model, mock_model, "Le modèle retourné n'est pas celui simulé."
        )

    def test_fetch_model_invalid_input(self):
        # Tester avec des entrées invalides
        with self.assertRaises(TypeError):
            fetch_model(
                REGISTERED_MODEL_NAME,
                REGISTERED_MODEL_ALIAS,
                "invalid_X_test",
                np.random.rand(4128),
            )

        with self.assertRaises(TypeError):
            fetch_model(
                REGISTERED_MODEL_NAME,
                REGISTERED_MODEL_ALIAS,
                np.random.rand(4128, 8),
                "invalid_y_test",
            )

    @patch("mlflow.pyfunc.load_model")
    def test_fetch_model_without_metrics(self, mock_load):
        # Simuler le chargement du modèle
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        # Cas 1 : X_test est None
        model, metrics = fetch_model(
            REGISTERED_MODEL_NAME,
            REGISTERED_MODEL_ALIAS,
            X_test=None,
            y_test=np.random.rand(4128),
        )
        self.assertEqual(
            model, mock_model, "Le modèle retourné n'est pas celui simulé."
        )
        self.assertIsNone(
            metrics,
            "Les métriques devraient être None lorsque X_test est None.",
        )

        # Cas 2 : y_test est None
        model, metrics = fetch_model(
            REGISTERED_MODEL_NAME,
            REGISTERED_MODEL_ALIAS,
            X_test=np.random.rand(4128, 8),
            y_test=None,
        )
        self.assertEqual(
            model, mock_model, "Le modèle retourné n'est pas celui simulé."
        )
        self.assertIsNone(
            metrics,
            "Les métriques devraient être None lorsque y_test est None.",
        )

        # Cas 3 : X_test et y_test sont None
        model, metrics = fetch_model(
            REGISTERED_MODEL_NAME,
            REGISTERED_MODEL_ALIAS,
            X_test=None,
            y_test=None,
        )
        self.assertEqual(
            model, mock_model, "Le modèle retourné n'est pas celui simulé."
        )
        self.assertIsNone(
            metrics,
            "Les métriques devraient être None lorsque X_test et y_test sont None.",
        )


if __name__ == "__main__":
    unittest.main()
