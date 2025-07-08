import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessors
model_data = joblib.load("fuel_predictor.pkl")
model = model_data['model']
scaler = model_data['scaler']
encoders = model_data['encoders']
features = model_data['features']

# Emission factors (tons CO2 / ton fuel)
emission_factors = {
    'HFO': 3.114,
    'MGO': 3.206,
    'LNG': 2.750
}

# CII grading thresholds (example, adjust if needed)
cii_grades = {
    'A': lambda x: x <= 5,
    'B': lambda x: 5 < x <= 10,
    'C': lambda x: 10 < x <= 15,
    'D': lambda x: 15 < x <= 20,
    'E': lambda x: x > 20
}

st.title("🚢 Ship Fuel Consumption & Emission Calculator")

# Inputs
ship_type = st.selectbox("Ship Type", encoders['ship_type'].classes_)
fuel_type = st.selectbox("Fuel Type", encoders['fuel_type'].classes_)
route_id = st.selectbox("Route ID", encoders['route_id'].classes_)
month = st.selectbox("Month", encoders['month'].classes_)
weather = st.selectbox("Weather Conditions", encoders['weather_conditions'].classes_)
distance = st.number_input("Distance Travelled (NM)", min_value=0.0, key="distance")
cargo_weight = st.number_input("Cargo Weight (tons)", min_value=0.0, key="cargo_weight")

if st.button("Predict Fuel Consumption & Calculate Emissions"):

    # Prepare input for prediction
    input_data = pd.DataFrame([{
        'ship_type': encoders['ship_type'].transform([ship_type])[0],
        'fuel_type': encoders['fuel_type'].transform([fuel_type])[0],
        'route_id': encoders['route_id'].transform([route_id])[0],
        'month': encoders['month'].transform([month])[0],
        'weather_conditions': encoders['weather_conditions'].transform([weather])[0],
        'distance': distance
    }])

    # Scale numeric features
    input_data[scaler.feature_names_in_] = scaler.transform(input_data[scaler.feature_names_in_])
    input_data = input_data[features]

    # Predict fuel consumption
    fuel_used = model.predict(input_data)[0]

    # Calculate emissions
    ef = emission_factors.get(fuel_type.upper(), 3.114)  # Default to HFO if not found
    co2_emitted = fuel_used * ef
    eeoi = co2_emitted / (cargo_weight * distance) if cargo_weight > 0 and distance > 0 else 0
    cii = (co2_emitted * 1_000_000) / (cargo_weight * distance) if cargo_weight > 0 and distance > 0 else 0

    # Determine CII grade
    grade = next((g for g, cond in cii_grades.items() if cond(cii)), "Unknown")

    # Display results
    st.success(f"📊 Predicted Fuel Consumption: **{fuel_used:.2f} MT**")
    st.info(f"🌿 Estimated CO₂ Emissions: **{co2_emitted:.2f} tons**")
    st.info(f"📉 EEOI: **{eeoi:.4f}**")
    st.info(f"📈 CII: **{cii:.2f} g/ton·NM** → Grade: **{grade}**")
