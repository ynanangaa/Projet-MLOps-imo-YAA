import unittest
from unittest.mock import patch, MagicMock
import mlflow
import mlflow.sklearn
from california_houseprice_prediction.domain.log_params_metrics_model import (
    log_parameters,
    log_metrics,
    log_model,
)


class TestLogParamsMetricsModel(unittest.TestCase):

    @patch("mlflow.log_param")
    def test_log_parameters(self, mock_log_param):
        # Données de test
        params = {"example_param": 42}

        # Appel de la fonction à tester
        log_parameters(params)

        # Vérifications
        self.assertEqual(mock_log_param.call_count, 1)
        mock_log_param.assert_called_once_with("example_param", 42)

    @patch("mlflow.log_metric")
    def test_log_metrics(self, mock_log_metric):
        # Données de test
        metrics = {"R2": 0.9, "RMSE": 0.1}

        # Appel de la fonction à tester
        log_metrics(metrics)

        # Vérifications
        self.assertEqual(mock_log_metric.call_count, 2)
        mock_log_metric.assert_any_call("R2", 0.9)
        mock_log_metric.assert_any_call("RMSE", 0.1)

    @patch("mlflow.sklearn.log_model")
    def test_log_model(self, mock_log_model):
        # Données de test
        model = MagicMock()  # Simuler un modèle

        # Appel de la fonction à tester
        log_model(model, "test-model")

        # Vérifications
        mock_log_model.assert_called_once_with(model, "test-model")


if __name__ == "__main__":
    unittest.main()
