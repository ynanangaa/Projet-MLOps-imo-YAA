import streamlit as st
import requests

st.title("House Price Prediction")

with st.form("house_price_form"):
    st.write("Enter housing features:")
    median_income = st.number_input("Median income (10K $)")
    house_age = st.number_input("Median house age")
    avg_rooms = st.number_input("Average number of rooms per household")
    avg_bedrooms = st.number_input("Average number of bedrooms per household")
    population = st.number_input("Population size")
    avg_occupancy = st.number_input("Average number of occupants per house")
    latitude = st.number_input("Latitude")
    longitude = st.number_input("Longitude")

    # Add more features as needed

    submitted = st.form_submit_button("Predict Median Price")
    if submitted:
        # Collect input data
        input_data = {
            "median_income": median_income,
            "house_age": house_age,
            "avg_rooms": avg_rooms,
            "avg_bedrooms": avg_bedrooms,
            "population": population,
            "avg_occupancy": avg_occupancy,
            "latitude": latitude,
            "longitude": longitude,
        }

        # Make API call to the prediction endpoint
        api_url = "http://127.0.0.1:8000/predict"
        response = requests.post(api_url, json=input_data)

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"The predicted house price is: ${prediction:.3f}")
        else:
            st.error("Failed to get prediction from API")
