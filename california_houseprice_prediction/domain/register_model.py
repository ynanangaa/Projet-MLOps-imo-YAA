import mlflow
from mlflow.tracking import MlflowClient
from config import REGISTERED_MODEL_ALIAS, REGISTERED_MODEL_NAME
from california_houseprice_prediction.infrastructure import EXPERIMENT_NAME


R2_MIN = 0.9
MAE_MAX = 0.33
RMSE_MAX = 0.49


def get_best_run_id(experiment_name=EXPERIMENT_NAME, R2_threshold=R2_MIN, MAE_threshold=MAE_MAX, RMSE_threshold=RMSE_MAX):
    """
    Récupère le run_id du meilleur modèle en fonction du MAE, parmi les modèles dont :
    - R2 > R2_threshold
    - MAE < MAE_threshold
    - RMSE < RMSE_threshold

    Args:
        experiment_name (str): Nom de l'expérience MLflow.
        R2_threshold (float): Seuil de R² pour filtrer les runs. Par défaut à R2_MIN.
        MAE_threshold (float): Seuil de MAE pour filtrer les runs. Par défaut à MAE_MAX.
        RMSE_threshold (float): Seuil de RMSE pour filtrer les runs. Par défaut à RMSE_MAX.

    Returns:
        str: run_id du meilleur modèle.
    """
    # Récupérer l'expérience
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Expérience '{experiment_name}' non trouvée.")

    # Filtrer les runs avec R2 > R2_threshold, MAE < MAE_threshold et RMSE < RMSE_threshold, puis trier par MAE croissant
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.RMSE < {RMSE_threshold} AND metrics.MAE < {MAE_threshold} AND metrics.R2 > {R2_threshold}",
        order_by=["metrics.MAE ASC"],  # Trier par MAE croissant
    )

    if runs.empty:
        return None

    else:
        # Trouver l'indice du run avec la RMSE la plus faible
        ind_min_RMSE = runs["metrics.RMSE"].idxmin()

        # Retourner le run_id du meilleur run
        best_run = runs.iloc[ind_min_RMSE]

        return best_run.run_id


def register_model(model_name, model_alias, experiment_name=EXPERIMENT_NAME, R2_threshold=R2_MIN, MAE_threshold=MAE_MAX, RMSE_threshold=RMSE_MAX):
    """
    Enregistre le meilleur modèle dans le MLflow Model Registry.

    Args:
        model_name (str): Nom du modèle dans le Registry.
        model_alias (str): Alias à attribuer au modèle.
        experiment_name (str): Nom de l'expérience MLflow.
        R2_threshold (float): Seuil de R² pour filtrer les runs. Par défaut à R2_MIN.
        MAE_threshold (float): Seuil de MAE pour filtrer les runs. Par défaut à MAE_MAX.
        RMSE_threshold (float): Seuil de RMSE pour filtrer les runs. Par défaut à RMSE_MAX.
    """

    # Récupérer le run_id du meilleur modèle
    run_id = get_best_run_id(experiment_name, R2_threshold, MAE_threshold, RMSE_threshold)

    # Si le run_id vaut None, on affiche :
    if run_id is None:
        print(f"Aucun run trouvé pour l'expérience '{EXPERIMENT_NAME}' avec R2 > {R2_MIN}, MAE < {MAE_MAX} et RMSE < {RMSE_MAX}.")
        print(f"Meilleur run_id trouvé : {run_id}")

    else:
        # Enregistrer le modèle dans le Registry
        print(f"Meilleur run_id trouvé : {run_id}")
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, model_name)
        print(
            f"Modèle enregistré avec succès : {result.name} (version {result.version})"
        )

        # Ajouter des tags et des alias
        client = MlflowClient()
        client.set_model_version_tag(
            model_name, result.version, "env", "production"
        )
        client.set_model_version_tag(
            model_name, result.version, "framework", "sklearn"
        )
        client.set_registered_model_alias(model_name, model_alias, result.version)

        print(
            f"Ajout des tags et de l'alias {model_alias} pour la version {result.version}."
        )


if __name__ == "__main__":
    # Enregistrer le meilleur modèle
    register_model(REGISTERED_MODEL_NAME, REGISTERED_MODEL_ALIAS)