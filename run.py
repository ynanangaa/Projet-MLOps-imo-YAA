from california_houseprice_prediction.infrastructure import load_and_split_data
from california_houseprice_prediction.domain import (
    train_and_log_base_model as train_linear_regression,
)
from california_houseprice_prediction.domain import (
    train_and_log_gradient_boosting_model as train_gradient_boosting,
)
from california_houseprice_prediction.domain import (
    train_and_log_random_forest_model as train_random_forest,
)

# Fixer le nombre d'estimateurs
n_estimators=120

def main():
    # Charger et diviser les données une seule fois
    print("Chargement et division des données...")
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Entraîner et logger le modèle de régression linéaire
    print("\nEntraînement du modèle de régression linéaire...")
    train_linear_regression(X_train, X_test, y_train, y_test)

    # Entraîner et logger le modèle de Gradient Boosting
    print("\nEntraînement des modèles de Gradient Boosting...")
    for learning_rate in [0.2, 0.3]:
        for max_depth in [3, 5, None]:
            for max_features in [3, None]:
                print(
                    f"\nGradient Boosting - learning_rate={learning_rate},"
                    f" n_estimators={n_estimators}, max_depth={max_depth},"
                    f" max_features={max_features}"
                )
                train_gradient_boosting(
                    X_train, X_test, y_train, y_test,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features
                )

    # Entraîner et logger le modèle de Random Forest
    print("\nEntraînement des modèles de Random Forest...")
    for max_depth in [5, 10, None]:
        for max_features in [3, None]:
            print(
                f"\nRandom Forest - max_depth={max_depth}, "
                f"max_features={max_features}, n_estimators={n_estimators}"
            )
            train_random_forest(
                X_train, X_test, y_train, y_test,
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features
            )

    print("\nTous les modèles ont été entraînés et enregistrés avec succès !")


if __name__ == "__main__":
    main()