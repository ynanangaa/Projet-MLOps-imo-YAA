from unittest.mock import patch, MagicMock
import mlflow
import mlflow.sklearn
from california_houseprice_prediction.domain.log_params_metrics_model import (
    log_parameters,
    log_metrics,
    log_model,
)


def test_log_parameters():
    with patch("mlflow.log_param") as mock_log_param:
        params = {"example_param": 42}
        log_parameters(params)

        assert mock_log_param.call_count == 1
        mock_log_param.assert_called_once_with("example_param", 42)


def test_log_metrics():
    with patch("mlflow.log_metric") as mock_log_metric:
        metrics = {"R2": 0.9, "RMSE": 0.1}
        log_metrics(metrics)

        assert mock_log_metric.call_count == 2
        mock_log_metric.assert_any_call("R2", 0.9)
        mock_log_metric.assert_any_call("RMSE", 0.1)


def test_log_model():
    with patch("mlflow.sklearn.log_model") as mock_log_model:
        model = MagicMock()  # Simuler un modèle
        log_model(model, "test-model")

        mock_log_model.assert_called_once_with(model, "test-model")
