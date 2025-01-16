# california_houseprice_prediction/domain/log_params_and_metrics.py

import mlflow
import mlflow.sklearn

def log_parameters(params):
    """
    Log les paramètres dans MLflow.
    
    Args:
        params (dict): Dictionnaire des paramètres à logger.
    """
    for key, value in params.items():
        mlflow.log_param(key, value)

def log_metrics(metrics):
    """
    Log les métriques dans MLflow.
    
    Args:
        metrics (dict): Dictionnaire des métriques à logger.
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

def log_model(model, artifact_name):
    """
    Log le modèle dans MLflow.
    
    Args:
        model: Modèle à logger.
        artifact_name (str): Nom de l'artéfact.
    """
    mlflow.sklearn.log_model(model, artifact_name)