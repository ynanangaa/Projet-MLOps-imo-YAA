import pytest
import streamlit.testing.v1 as st_test
import re


def test_streamlit_interface():
    # Charge l'application Streamlit
    app = st_test.AppTest.from_file(
        "california_houseprice_prediction/interface/streamlit_interface.py"
    ).run()

    # Assertion 1 : Vérifie que le titre est correct
    assert app.title[0].value == "House Price Prediction"

    # Assertion 2 : Vérifie que les textes des zones de saisie sont corrects
    assert (
        "Median income (10K $)" in app.number_input(key="median_income").label
    )
    assert "Median house age" in app.number_input(key="house_age").label
    assert (
        "Average number of rooms per household"
        in app.number_input(key="avg_rooms").label
    )
    assert (
        "Average number of bedrooms per household"
        in app.number_input(key="avg_bedrooms").label
    )
    assert "Population" in app.number_input(key="population").label
    assert (
        "Average household occupancy"
        in app.number_input(key="avg_occupancy").label
    )
    assert "Latitude" in app.number_input(key="latitude").label
    assert "Longitude" in app.number_input(key="longitude").label

    # Remplit les champs en utilisant les "key"
    app.number_input(key="median_income").set_value(8.3252).run()
    app.number_input(key="house_age").set_value(41.0).run()
    app.number_input(key="avg_rooms").set_value(6.984127).run()
    app.number_input(key="avg_bedrooms").set_value(1.023810).run()
    app.number_input(key="population").set_value(322.0).run()
    app.number_input(key="avg_occupancy").set_value(2.555556).run()
    app.number_input(key="latitude").set_value(37.88).run()
    app.number_input(key="longitude").set_value(-122.23).run()

    # Clique sur le bouton "Predict Price"
    app.button[0].click().run()

    # Assertion 3 : Vérifie que la prédiction est affichée
    assert "The predicted house price is:" in app.success[0].value

    # Assertion 4 : Vérifie que la prédiction est un nombre valide
    predicted_price = app.success[0].value.split(":")[1].strip()
    predicted_price = re.sub(
        r"[^0-9.]", "", predicted_price
    )  # Supprime tout sauf les chiffres et le point

    try:
        float(predicted_price)  # Convertit la prédiction en nombre
        assert True  # Si la conversion réussit, la prédiction est valide
    except ValueError:
        pytest.fail("La prédiction n'est pas un nombre valide.")
