import pytest
from california_houseprice_prediction.infrastructure import load_and_split_data


def test_load_and_split_data():
    X_train, X_test, y_train, y_test = load_and_split_data(
        test_size=0.2, random_state=42
    )

    # Vérifier que les données sont bien divisées
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

    # Vérifier que la taille de l'ensemble de test est correcte
    assert len(X_test) / (len(X_train) + len(X_test)) == pytest.approx(0.2)
