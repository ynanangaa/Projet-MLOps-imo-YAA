import mlflow
from mlflow.tracking import MlflowClient

def register_model(run_id, model_name):
    """
    Enregistre un modèle dans le MLflow Model Registry.

    Args:
        run_id (str): ID de la run MLflow où le modèle est loggé.
        model_name (str): Nom du modèle dans le Registry.
    """
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
    # Exemple : Enregistrer un modèle spécifique
    run_id = "ff2f99b235ea45e6b687b97addd00a98"  # Remplace par l'ID de ta run
    model_name = "sk-learn-gradient-boosting-reg"
    register_model(run_id, model_name)