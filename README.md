# Projet MLOps - Prédiction des prix des maisons en Californie

Ce projet vise à prédire les prix des maisons en Californie en utilisant des modèles de machine learning. Il est structuré pour faciliter l'expérimentation, l'enregistrement des modèles, et le déploiement via une API.

## Structure du projet

Le projet est organisé en plusieurs modules pour une séparation claire des responsabilités :

- **`infrastructure/`** : Charge et divise les données (`load_and_split_data.py`).
- **`domain/`** : Contient la logique métier, y compris l'entraînement des modèles (`train_and_log_*.py`) et l'enregistrement des modèles (`register_model.py`).
- **`application/`** : Gère l'API (`api.py`), la récupération des modèles (`fetch_model.py`), et le serving MLflow (`mlflow_model_serving.py`).
- **`interface/`** : Interface utilisateur avec Streamlit (`streamlit_interface.py`).

Le dossier **`tests/`** suit la même structure pour les tests unitaires.

## Configuration

Le fichier **`config.py`** contient les variables `REGISTERED_MODEL_NAME` et `REGISTERED_MODEL_ALIAS` pour enregistrer les modèles dans MLflow Registry.

## Installation et utilisation

1. **Cloner le projet** :
   ```bash
   git clone https://github.com/ynanangaa/Projet-MLOps-imo-YAA.git
   cd Projet-MLOps-imo-YAA
   ```

2. **Configurer l'environnement** :
   Exécutez **`activate.py`** pour installer Poetry et configurer l'environnement :
   ```bash
   python activate.py
   ```

3. **Lancer les expérimentations** :
   Entraînez et enregistrez les modèles avec :
   ```bash
   poetry run python run.py
   poetry run mlflow ui
   ```

4. **Enregistrer le meilleur modèle** :
   ```bash
   poetry run python california_houseprice_prediction/domain/register_model.py
   ```

5. **Récupérer le modèle** :
   ```bash
   poetry run python california_houseprice_prediction/application/fetch_model.py
   ```

6. **Démarrer l'API** :
   ```bash
   poetry run uvicorn california_houseprice_prediction.application.api:app --reload
   ```

7. **Servir le modèle avec MLflow** :
   ```bash
   poetry run mlflow models serve -m "models:/[REGISTERED_MODEL_NAME]@[REGISTERED_MODEL_ALIAS]" -p 5001 --no-conda
   poetry run python california_houseprice_prediction/application/mlflow_model_serving.py
   ```

## Tests

Les tests unitaires sont organisés dans le dossier **`tests/`**, suivant la même structure que le projet principal.

- **Tests Pytest** :
  ```bash
  poetry run pytest tests/ -v
  ```

- **Tests Unittest** :
  ```bash
  poetry run python -m unittest discover tests/ -v
  ```
