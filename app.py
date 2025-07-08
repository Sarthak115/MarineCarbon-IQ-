import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and preprocessors
model_data = joblib.load("fuel_predictor.pkl")
model = model_data['model']
scaler = model_data['scaler']
encoders = model_data['encoders']
features = model_data['features']

st.set_page_config(page_title="Ship Fuel Estimator", layout="wide")
st.title("🚢 Ship Fuel Consumption Estimator")

# UI Inputs
ship_type = st.selectbox("Ship Type", encoders['ship_type'].classes_)
fuel_type = st.selectbox("Fuel Type", encoders['fuel_type'].classes_)
route_id = st.selectbox("Route ID", encoders['route_id'].classes_)
month = st.selectbox("Month", encoders['month'].classes_)
weather = st.selectbox("Weather Conditions", encoders['weather_conditions'].classes_)
distance = st.number_input("Distance Travelled (NM)", min_value=0.0)
cargo_weight = st.number_input("Cargo Weight (tons)", min_value=1.0)

# Prediction
if st.button("🚀 Predict Fuel Consumption"):
    input_data = pd.DataFrame([{
        'ship_type': encoders['ship_type'].transform([ship_type])[0],
        'fuel_type': encoders['fuel_type'].transform([fuel_type])[0],
        'route_id': encoders['route_id'].transform([route_id])[0],
        'month': encoders['month'].transform([month])[0],
        'weather_conditions': encoders['weather_conditions'].transform([weather])[0],
        'distance': distance
    }])

    input_data[scaler.feature_names_in_] = scaler.transform(input_data[scaler.feature_names_in_])
    input_data = input_data[features]
    prediction = model.predict(input_data)[0]
    st.success(f"📊 Predicted Fuel Consumption: **{prediction:.2f} MT**")

    # Emission metrics
    ef_map = {'HFO': 3.114, 'MGO': 3.206, 'LNG': 2.750}
    ef = ef_map.get(fuel_type, 3.114)
    co2 = prediction * ef
    eeoi = co2 / (cargo_weight * distance) if cargo_weight * distance else 0
    cii = (co2 * 1_000_000) / (cargo_weight * distance) if cargo_weight * distance else 0
    grade = 'A' if cii <= 2 else 'B' if cii <= 4 else 'C' if cii <= 6 else 'D' if cii <= 8 else 'E'

    st.markdown("### 🧮 Emission Metrics")
    st.write(f"**CO₂ Emitted:** {co2:.2f} tons")
    st.write(f"**EEOI:** {eeoi:.4f}")
    st.write(f"**CII:** {cii:.4f} → Grade: **{grade}**")

    # Append to CSV
    csv_path = "ship_fuel_efficiency.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'cii' not in df.columns: df['cii'] = None
        if 'cii_grade' not in df.columns: df['cii_grade'] = None
    else:
        df = pd.DataFrame()

    new_row = {
        'ship_type': ship_type,
        'fuel_type': fuel_type,
        'route_id': route_id,
        'month': month,
        'weather_conditions': weather,
        'distance': distance,
        'fuel_consumption': prediction,
        'co2_emissions': co2,
        'cii': cii,
        'cii_grade': grade
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    st.success("📁 Logged entry to CSV.")

    # ------------------- DASHBOARD -------------------

    st.markdown("## 📈 Visual Analytics")

    # Columns: Left (Details + AI), Right (Charts)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📋 Current Input Summary")
        st.write(f"**Ship Type:** {ship_type}")
        st.write(f"**Fuel Type:** {fuel_type}")
        st.write(f"**Route ID:** {route_id}")
        st.write(f"**Month:** {month}")
        st.write(f"**Weather:** {weather}")
        st.write(f"**Distance:** {distance} NM")
        st.write(f"**Cargo Weight:** {cargo_weight} tons")
        st.write(f"**Predicted Fuel:** {prediction:.2f} MT")
        st.write(f"**CII Grade:** {grade}")

        st.markdown("---")
        st.markdown("### 🤖 Agentic AI Suggestions")
        st.info("Agentic AI recommendations will appear here...")

    with col2:
        st.markdown("### 📊 Performance Overview")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            # Prepare missing data
            ef_map = {'HFO': 3.114, 'MGO': 3.206, 'LNG': 2.750}
            if 'co2_emissions' not in df.columns:
                df['co2_emissions'] = df['fuel_consumption'] * df['fuel_type'].map(ef_map)
            if 'cii' not in df.columns:
                df['cii'] = (df['co2_emissions'] * 1_000_000) / (10000 * df['distance'])
            if 'cii_grade' not in df.columns:
                df['cii_grade'] = pd.cut(df['cii'], bins=[0, 2, 4, 6, 8, float('inf')], labels=list("ABCDE"))

            # Organize into 2x2 grid
            top1, top2 = st.columns(2)
            bottom1, bottom2 = st.columns(2)

            with top1:
                fig1, ax1 = plt.subplots(figsize=(4, 3))
                sns.scatterplot(data=df, x='distance', y='fuel_consumption', hue='ship_type', ax=ax1, s=30)
                ax1.set_title("Fuel vs Distance", fontsize=10)
                ax1.tick_params(labelsize=8)
                st.pyplot(fig1, use_container_width=True)

            with top2:
                fig2, ax2 = plt.subplots(figsize=(4, 3))
                grouped = df.dropna(subset=["ship_type", "fuel_consumption"]).groupby('ship_type')['fuel_consumption'].mean().sort_values()
                if not grouped.empty:
                  grouped.plot(kind='barh', ax=ax2)
                  ax2.set_title("Avg Fuel by Ship Type", fontsize=10)
                else:
                   ax2.text(0.5, 0.5, "No valid data to display", ha='center', va='center', fontsize=9)
                ax2.tick_params(labelsize=8)
                st.pyplot(fig2, use_container_width=True)


            with bottom1:
                fig3, ax3 = plt.subplots(figsize=(4, 3))
                df.groupby('month')['co2_emissions'].sum().plot(kind='line', marker='o', ax=ax3)
                ax3.set_title("Monthly CO₂ Emissions", fontsize=10)
                ax3.tick_params(labelsize=8)
                st.pyplot(fig3, use_container_width=True)

            with bottom2:
                fig4, ax4 = plt.subplots(figsize=(4, 3))
                sns.countplot(data=df, x='cii_grade', order=list("ABCDE"), ax=ax4, palette='muted')
                ax4.set_title("CII Grade Distribution", fontsize=10)
                ax4.tick_params(labelsize=8)
                st.pyplot(fig4, use_container_width=True)

        else:
            st.warning("No data available yet for visualization.")
