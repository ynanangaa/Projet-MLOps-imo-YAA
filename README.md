# Projet MLOps - Californie House Prices

Ce dépôt contient une application MLOps conçue pour entraîner, versionner et servir un modèle de prédiction de prix de maisons en Californie. Le projet est structuré pour faciliter le développement local et le déploiement Docker, avec une API FastAPI et une interface Streamlit.

## Structure du projet

Le code est organisé en modules logiques, pour garder une séparation claire entre entraînement, service et interface.

- **`src/`** : logique principale de l’application. C’est la base du package Python.
- **`ui/`** : interface utilisateur Streamlit et API FastAPI.
- **`tests/`** : suites de tests unitaires et d’intégration.
- **`models/`** : contiendra les modèles sauvegardés et/ou extraits par MLflow.
- **`notebooks/`** : expérimentations et analyses exploratoires.
- **`mlruns/`** : répertoire MLflow pour les exécutions locales.
- **`Dockerfile`** : image Docker optimisée pour exécuter l’application avec `uv`.
- **`compose.yaml`** : orchestration Docker Compose pour les services `train`, `api` et `streamlit`.


## Commandes Docker

### 1. Construction de l’image

```bash
docker compose build
```

### 2. Lancement des services

```bash
docker compose up -d
```

Cela démarre :

- Le service `train` qui exécute l’entraînement du modèle
- `http://localhost:8000` pour l’API
- `http://localhost:8501` pour l’interface Streamlit

### 3. Arrêt

```bash
docker compose down
```

### 4. Recréer l’environnement proprement

```bash
docker compose down --volumes
docker compose up --build
```

## Exécution locale sans Docker

Si vous préférez rester sur un environnement local Python, le projet utilise `pyproject.toml` et `uv`.

### Installation des dépendances

```bash
python -m pip install -U pip
python -m pip install uv
uv sync
```

### Lancer l’entraînement localement

```bash
uv run src/train.py
```

### Lancer l’API localement

```bash
uv run uvicorn ui.api:app --host 0.0.0.0 --port 8000 --reload
```

### Lancer l’interface Streamlit localement

```bash
uv run streamlit run ui/prediction_ui.py --server.address=0.0.0.0
```

### Visualiser les expérimentations MLflow

Pour visualiser les expérimentations dans MLflow, lancez l’interface utilisateur :

```bash
uv run mlflow ui
```

Cela ouvrira l’interface à `http://localhost:5000`.

## Points d’entrée et dossiers clés

- **API** : `ui/api.py`
- **Interface utilisateur** : `ui/prediction_ui.py`
- **Modèles** : `models/`
- **Expérimentations MLflow** : `mlruns/`

## Liens utiles

- FastAPI : https://fastapi.tiangolo.com/
- Streamlit : https://docs.streamlit.io/
- Docker Compose : https://docs.docker.com/compose/
- MLflow : https://mlflow.org/docs/latest/index.html

## Notes

- Cette documentation suppose que vous connaissez déjà Docker, Python et les concepts de base de l’architecture MLOps.
- Le dossier `ui/` est le point d’accès principal pour le service et l’interface utilisateur.
- `compose.yaml` est le fichier de référence pour l’orchestration Docker de l’application.
