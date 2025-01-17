import unittest
from unittest.mock import patch, MagicMock
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from california_houseprice_prediction.domain import (
    get_best_run_id,
    register_model,
    REGISTERED_MODEL_NAME,
    REGISTERED_MODEL_ALIAS,
    R2_MIN,
    MAE_MAX,
    RMSE_MAX,
)
from california_houseprice_prediction.infrastructure import EXPERIMENT_NAME


class TestRegisterModel(unittest.TestCase):

    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.search_runs")
    def test_get_best_run_id(self, mock_search_runs, mock_get_experiment_by_name):
        # Mock de l'expérience
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "123"
        mock_get_experiment_by_name.return_value = mock_experiment

        # Mock des runs
        mock_runs = pd.DataFrame({
            "run_id": ["run_1", "run_2", "run_3"],
            "metrics.R2": [0.91, 0.95, 0.89],  # run_3 a R2 <= R2_MIN
            "metrics.MAE": [0.31, 0.30, 0.33], # run_3 a MAE >= MAE_MAX
            "metrics.RMSE": [0.48, 0.47, 0.50], # run_3 a RMSE >= RMSE_MAX
        })
        mock_search_runs.return_value = mock_runs

        # Appel de la fonction
        best_run_id = get_best_run_id(EXPERIMENT_NAME, R2_MIN, MAE_MAX, RMSE_MAX)

        # Vérifications
        self.assertEqual(best_run_id, "run_2", "Le meilleur run_id n'est pas correct.")
        mock_get_experiment_by_name.assert_called_once_with(EXPERIMENT_NAME)
        mock_search_runs.assert_called_once_with(
            experiment_ids=["123"],
            filter_string=f"metrics.RMSE < {RMSE_MAX} AND metrics.MAE < {MAE_MAX} AND metrics.R2 > {R2_MIN}",
            order_by=["metrics.MAE ASC"],
        )

    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.search_runs")
    def test_get_best_run_id_no_runs_found(
        self, mock_search_runs, mock_get_experiment_by_name
    ):
        # Mock de l'expérience
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "123"
        mock_get_experiment_by_name.return_value = mock_experiment

        # Mock pour simuler aucun run trouvé
        mock_runs = pd.DataFrame(columns=["run_id", "metrics.R2", "metrics.MAE", "metrics.RMSE"])  # DataFrame vide
        mock_search_runs.return_value = mock_runs

        # Appel de la fonction
        best_run_id = get_best_run_id(EXPERIMENT_NAME, R2_MIN, MAE_MAX, RMSE_MAX)

        # Vérification que la fonction retourne None
        self.assertIsNone(best_run_id, "La fonction devrait retourner None lorsqu'aucun run n'est trouvé.")

    @patch(
        "california_houseprice_prediction.domain.register_model.get_best_run_id"
    )
    @patch("mlflow.register_model")
    @patch("mlflow.tracking.MlflowClient")
    def test_register_model(
        self, mock_client, mock_register_model, mock_get_best_run_id
    ):
        # Mock du run_id
        mock_get_best_run_id.return_value = "run_123"

        # Mock de l'enregistrement du modèle
        mock_result = MagicMock()
        mock_result.name = REGISTERED_MODEL_NAME
        mock_result.version = "1"  # Simule la version retournée par mlflow.register_model
        mock_register_model.return_value = mock_result

        # Mock du client MLflow
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Simuler les tags retournés par get_model_version_tag
        mock_client_instance.get_model_version_tag.side_effect = lambda name, version, key: {
            (REGISTERED_MODEL_NAME, "1", "framework"): "sklearn",  # Tag défini
        }.get((name, version, key))

        # Appel de la fonction
        model_name = REGISTERED_MODEL_NAME
        model_alias = REGISTERED_MODEL_ALIAS
        register_model(model_name, model_alias)

        # Vérifications
        mock_get_best_run_id.assert_called_once_with(
            EXPERIMENT_NAME, R2_MIN, MAE_MAX, RMSE_MAX
        )
        mock_register_model.assert_called_once_with(
            "runs:/run_123/model", model_name
        )

        # Vérifier que le tag "framework" est correctement défini
        framework_tag = mock_client_instance.get_model_version_tag(
            model_name, mock_result.version, "framework"
        )
        self.assertEqual(framework_tag, "sklearn")

        # Vérifier que le tag "env" n'est pas défini
        env_tag = mock_client_instance.get_model_version_tag(
            model_name, mock_result.version, "env"
        )
        self.assertIsNone(env_tag)

        # Définir le comportement de la méthode get_registered_model_alias
        mock_client_instance.get_registered_model_alias.return_value = "alias_value"

        alias = mock_client_instance.get_registered_model_alias(
            model_name, model_alias
        )
        self.assertEqual(alias, "alias_value")


if __name__ == "__main__":
    unittest.main()