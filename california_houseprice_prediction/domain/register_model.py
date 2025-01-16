import mlflow
from mlflow.tracking import MlflowClient

def get_best_run_id(experiment_name, metric_name="mae", order="ASC"):
    """
    Récupère le run_id du meilleur modèle en fonction d'une métrique.

    Args:
        experiment_name (str): Nom de l'expérience MLflow.
        metric_name (str): Nom de la métrique à utiliser pour la sélection (par défaut : "mae").
        order (str): Ordre de tri ("ASC" pour ascending, "DESC" pour descending).

    Returns:
        str: run_id du meilleur modèle.
    """
    # Récupérer l'expérience
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Expérience '{experiment_name}' non trouvée.")

    # Récupérer les runs de l'expérience
    runs = mlflow.search_runs(experiment.experiment_id, order_by=[f"metrics.{metric_name} {order}"])
    if runs.empty:
        raise ValueError(f"Aucun run trouvé pour l'expérience '{experiment_name}'.")

    # Retourner le run_id du meilleur modèle
    best_run = runs.iloc[0]
    return best_run.run_id

def register_model(model_name, experiment_name="California Housing"):
    """
    Enregistre le meilleur modèle dans le MLflow Model Registry.

    Args:
        model_name (str): Nom du modèle dans le Registry.
        experiment_name (str): Nom de l'expérience MLflow.
    """
    # Récupérer le run_id du meilleur modèle
    run_id = get_best_run_id(experiment_name)
    print(f"Meilleur run_id trouvé : {run_id}")

    # Enregistrer le modèle dans le Registry
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    print(f"Modèle enregistré avec succès : {result.name} (version {result.version})")

    # Ajouter des tags et des alias
    client = MlflowClient()
    client.set_model_version_tag(model_name, result.version, "env", "production")
    client.set_model_version_tag(model_name, result.version, "framework", "sklearn")
    client.set_registered_model_alias(model_name, "champion", result.version)

    print(f"Ajout des tags et de l'alias 'champion' pour la version {result.version}.")

if __name__ == "__main__":
    # Enregistrer le meilleur modèle
    model_name = "sk-learn-gradient-boosting-reg"
    register_model(model_name)