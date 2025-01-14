from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_and_split_data(test_size=0.2, random_state=42):
    """
    Charge les données california housing et les divise en ensembles d'entraînement et de test.

    Args:
        test_size (float): Proportion des données à utiliser pour le test.
        random_state (int): Seed pour la reproductibilité.

    Returns:
        X_train, X_test, y_train, y_test: Données divisées.
    """
    data = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
