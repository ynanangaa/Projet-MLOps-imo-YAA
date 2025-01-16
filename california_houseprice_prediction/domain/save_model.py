import mlflow

# Rechercher le modèle enregistré sous le nom "best_model"
client = mlflow.tracking.MlflowClient()
model_version = client.get_latest_versions("best_model", stages=["None"])[0]
run_id = model_version.run_id

# Enregistrer le modèle dans le Model Registry
model_name = "best_model"
model_version = client.create_model_version(
    name=model_name,
    source=f"runs:/{run_id}/model",  # Chemin vers le modèle dans MLflow
    run_id=run_id
)

# Transitionner la version du modèle vers l'étape de production
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production"
)

print("Modèle bien enregistré dans MLflow Registry.")