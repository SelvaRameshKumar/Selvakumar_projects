# streamlit_app_narrow_range.py  
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load Model and Columns
# -----------------------------
rf_model = joblib.load('car_price_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# -----------------------------
# App Title
# -----------------------------
st.title("CarDekho Used Car Price Predictor")
st.write("Enter the car details to get an estimated price range:")

# -----------------------------
# User Input
# -----------------------------
# Numeric Inputs
kms = st.number_input("Kms Driven", min_value=0, max_value=500000, value=50000, step=1000)
mileage = st.number_input("Mileage (kmpl or km/kg)", min_value=5.0, max_value=50.0, value=18.0, step=0.1)
engine_disp = st.number_input("Engine Displacement (cc)", min_value=500, max_value=5000, value=1200, step=10)
max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=500.0, value=70.0, step=0.1)
torque = st.number_input("Torque (Nm)", min_value=10.0, max_value=500.0, value=90.0, step=0.1)
seats = st.number_input("Seats", min_value=2, max_value=10, value=5, step=1)
car_age = st.number_input("Car Age (years)", min_value=0, max_value=30, value=5, step=1)

# Categorical Inputs
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'Electric', 'LPG'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
oem = st.selectbox("OEM", ['Maruti', 'Hyundai', 'Toyota', 'Honda', 'Mahindra', 'Kia', 'Ford', 'Tata', 'Volkswagen', 'Renault', 'Other'])
model_name = st.selectbox("Model", ['Swift', 'i20', 'Fortuner', 'City', 'Bolero', 'Seltos', 'EcoSport', 'Nexon', 'Vento', 'Duster', 'Other'])
city = st.selectbox("City", ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Kolkata', 'Pune', 'Other'])

# -----------------------------
# Prepare Input for Model
# -----------------------------
def preprocess_input():
    input_dict = {
        'Kms Driven': kms,
        'Mileage': mileage,
        'Engine Displacement': engine_disp,
        'Max Power': max_power,
        'Torque': torque,
        'Seats': seats,
        'car_age': car_age,
        'Fuel Type_'+fuel_type: 1,
        'Transmission_'+transmission: 1,
        'oem_'+oem: 1,
        'model_'+model_name: 1,
        'city_'+city: 1
    }

    df = pd.DataFrame([input_dict])

    # Add missing columns with 0
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df[model_columns]
    return df

# -----------------------------
# Prediction with Narrow Price Range
# -----------------------------
if st.button("Predict Price Range"):
    input_df = preprocess_input()

    # Get predictions from all trees
    all_tree_preds = np.array([tree.predict(input_df)[0] for tree in rf_model.estimators_])

    # Median prediction
    median = np.median(all_tree_preds)

    # Narrow range: ±20% of median
    lower = max(median * 0.8, 0)
    upper = median * 1.2

    st.success(f"Estimated Price: ₹ {median:,.0f}")
    st.info(f"Price Range: ₹ {lower:,.0f} – ₹ {upper:,.0f}")


