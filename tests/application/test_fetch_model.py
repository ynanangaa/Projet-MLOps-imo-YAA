import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from california_houseprice_prediction.application.fetch_model import (
    fetch_model,
)


def test_fetch_model():
    with patch("mlflow.pyfunc.load_model") as mock_load:
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
            "sk-learn-gradient-boosting-reg",
            "champion",
            mock_X_test,
            mock_y_test,
        )

        # Vérifier que mlflow.pyfunc.load_model a été appelé avec les bons arguments
        mock_load.assert_called_once_with(
            "models:/sk-learn-gradient-boosting-reg@champion"
        )

        # Vérifier que model.predict a été appelé avec mock_X_test
        mock_model.predict.assert_called_once_with(mock_X_test)

        # Vérifier que les métriques sont retournées
        assert "RMSE" in metrics, "La métrique RMSE n'est pas retournée."
        assert "MAE" in metrics, "La métrique MAE n'est pas retournée."

        # Vérifier que le modèle retourné est bien celui simulé
        assert (
            model == mock_model
        ), "Le modèle retourné n'est pas celui simulé."


def test_fetch_model_invalid_input():
    # Tester avec des entrées invalides
    with pytest.raises(TypeError):
        fetch_model(
            "sk-learn-gradient-boosting-reg",
            "champion",
            "invalid_X_test",
            np.random.rand(4128),
        )

    with pytest.raises(TypeError):
        fetch_model(
            "sk-learn-gradient-boosting-reg",
            "champion",
            np.random.rand(4128, 8),
            "invalid_y_test",
        )
