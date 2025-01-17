import unittest
from unittest.mock import patch, MagicMock
import mlflow
from mlflow.tracking import MlflowClient
from california_houseprice_prediction.domain.register_model import get_best_run_id, register_model

class TestRegisterModel(unittest.TestCase):

    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.search_runs")
    def test_get_best_run_id(self, mock_search_runs, mock_get_experiment_by_name):
        # Mock de l'expérience
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "123"
        mock_get_experiment_by_name.return_value = mock_experiment

        # Mock des runs
        mock_runs = MagicMock()
        mock_runs.empty = False
        mock_runs.iloc = [MagicMock(run_id="run_123")]
        mock_search_runs.return_value = mock_runs

        # Appel de la fonction
        run_id = get_best_run_id(experiment_name="california-housing", metric_name="mae", order="ASC")

        # Vérifications
        self.assertEqual(run_id, "run_123")
        mock_get_experiment_by_name.assert_called_once_with("california-housing")
        mock_search_runs.assert_called_once_with("123", order_by=["metrics.mae ASC"])

    @patch("register_model.get_best_run_id")
    @patch("mlflow.register_model")
    @patch("mlflow.tracking.MlflowClient")
    def test_register_model(self, mock_client, mock_register_model, mock_get_best_run_id):
        # Mock du run_id
        mock_get_best_run_id.return_value = "run_123"

        # Mock de l'enregistrement du modèle
        mock_result = MagicMock()
        mock_result.name = "sk-learn-gradient-boosting-reg"
        mock_result.version = "1"
        mock_register_model.return_value = mock_result

        # Mock du client MLflow
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Appel de la fonction
        model_name = "sk-learn-gradient-boosting-reg"
        register_model(model_name)

        # Vérifications
        mock_get_best_run_id.assert_called_once_with(experiment_name="california-housing")
        mock_register_model.assert_called_once_with("runs:/run_123/model", model_name)
        mock_client_instance.set_model_version_tag.assert_any_call(model_name, "1", "env", "production")
        mock_client_instance.set_model_version_tag.assert_any_call(model_name, "1", "framework", "sklearn")
        mock_client_instance.set_registered_model_alias.assert_called_once_with(model_name, "champion", "1")

    @patch("mlflow.get_experiment_by_name")
    def test_get_best_run_id_experiment_not_found(self, mock_get_experiment_by_name):
        # Mock pour simuler une expérience non trouvée
        mock_get_experiment_by_name.return_value = None

        # Vérification que l'exception est levée
        with self.assertRaises(ValueError) as context:
            get_best_run_id(experiment_name="non-existent-experiment")
        self.assertEqual(str(context.exception), "Expérience 'non-existent-experiment' non trouvée.")

    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.search_runs")
    def test_get_best_run_id_no_runs_found(self, mock_search_runs, mock_get_experiment_by_name):
        # Mock de l'expérience
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "123"
        mock_get_experiment_by_name.return_value = mock_experiment

        # Mock pour simuler aucun run trouvé
        mock_runs = MagicMock()
        mock_runs.empty = True
        mock_search_runs.return_value = mock_runs

        # Vérification que l'exception est levée
        with self.assertRaises(ValueError) as context:
            get_best_run_id(experiment_name="california-housing")
        self.assertEqual(str(context.exception), "Aucun run trouvé pour l'expérience 'california-housing'.")

if __name__ == "__main__":
    unittest.main()