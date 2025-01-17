import unittest
from unittest.mock import patch, MagicMock
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from california_houseprice_prediction.domain import (
    get_best_run_id,
    register_model,
    R2_MIN,
    MAE_MAX,
    RMSE_MAX,
)
from config import REGISTERED_MODEL_NAME, REGISTERED_MODEL_ALIAS
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


if __name__ == "__main__":
    unittest.main()