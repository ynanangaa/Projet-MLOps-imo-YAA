# run.py
from california_houseprice_prediction.infrastructure.split_data_train_test \
    import (load_and_split_data,)
from california_houseprice_prediction.domain.train_base_model import (
    train_and_log_base_model as train_linear_regression,
)
from california_houseprice_prediction.domain.train_gradient_boosting_model \
    import (train_and_log_gradient_boosting_model as train_gradient_boosting,)
from california_houseprice_prediction.domain.train_random_forest_model import (
    train_and_log_random_forest_model as train_random_forest,
)


def main():
    # Charger et diviser les données une seule fois
    print("Chargement et division des données...")
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Entraîner et logger le modèle de régression linéaire
    print("\nEntraînement du modèle de régression linéaire...")
    train_linear_regression(X_train, X_test, y_train, y_test)

    # Entraîner et logger le modèle de Gradient Boosting
    print("\nEntraînement du modèle de Gradient Boosting...")
    train_gradient_boosting(X_train, X_test, y_train, y_test)

    # Entraîner et logger le modèle de Random Forest
    print("\nEntraînement du modèle de Random Forest...")
    train_random_forest(X_train, X_test, y_train, y_test)

    print("\nTous les modèles ont été entraînés et enregistrés avec succès !")


if __name__ == "__main__":
    main()
