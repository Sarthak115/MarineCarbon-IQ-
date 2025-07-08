import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🚢 Fuel Estimator & Dashboard", layout="wide")

# Load model & components
model_data = joblib.load("fuel_predictor.pkl")
model = model_data['model']
scaler = model_data['scaler']
encoders = model_data['encoders']
features = model_data['features']

# Load dataset for graphs
try:
    df = pd.read_csv("ship_fuel_efficiency.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df = df.dropna()
except:
    st.warning("Could not load dataset for visualizations.")
    df = None

st.title("🚢 Ship Fuel Consumption Estimator")

# --- Input Section ---
col1, col2 = st.columns(2)
with col1:
    ship_type = st.selectbox("Ship Type", encoders['ship_type'].classes_)
    fuel_type = st.selectbox("Fuel Type", encoders['fuel_type'].classes_)
    route_id = st.selectbox("Route ID", encoders['route_id'].classes_)
with col2:
    month = st.selectbox("Month", encoders['month'].classes_)
    weather = st.selectbox("Weather Conditions", encoders['weather_conditions'].classes_)
    distance = st.number_input("Distance Travelled (NM)", min_value=0.0, key="distance_travelled")

cargo_weight = st.number_input("Cargo Weight (tons)", min_value=1.0, value=10000.0)

if st.button("Predict Fuel Consumption"):

    # Encode and scale input
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

    # --- Calculations ---
    emission_factors = {'HFO': 3.114, 'MGO': 3.206, 'LNG': 2.750}
    ef = emission_factors.get(fuel_type, 3.0)
    co2_emitted = prediction * ef
    eeoi = co2_emitted / (cargo_weight * distance) if distance else 0
    cii = (co2_emitted * 1_000_000) / (cargo_weight * distance) if distance else 0

    # Grade mapping (example thresholds)
    if cii < 4:
        grade = 'A'
    elif cii < 6:
        grade = 'B'
    elif cii < 9:
        grade = 'C'
    elif cii < 12:
        grade = 'D'
    else:
        grade = 'E'

    # --- Output ---
    st.success(f"📊 Predicted Fuel Consumption: **{prediction:.2f} MT**")
    st.info(f"🌿 CO₂ Emitted: **{co2_emitted:.2f} tons**")
    st.info(f"📈 EEOI: **{eeoi:.4f}**")
    st.info(f"⚙️ CII: **{cii:.2f} g/ton·NM** → Grade **{grade}**")

    # --- Dashboard Section ---
    st.markdown("---")
    st.subheader("📊 Visual Dashboard")

    if df is not None:
        # Preprocess if not already done
        if 'co2_emissions' not in df.columns:
            ef_map = {'HFO': 3.114, 'MGO': 3.206, 'LNG': 2.750}
            df['ef'] = df['fuel_type'].map(ef_map)
            df['co2_emissions'] = df['fuel_consumption'] * df['ef']
            df['cii'] = (df['co2_emissions'] * 1_000_000) / (10000 * df['distance'])
            df['cii_grade'] = pd.cut(df['cii'], bins=[0,4,6,9,12,float('inf')], labels=list("ABCDE"))

        # Layout
        g1, g2 = st.columns(2)

        with g1:
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='distance', y='fuel_consumption', hue='ship_type', ax=ax)
            plt.title("Fuel Consumption vs Distance")
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            avg_fuel = df.groupby('ship_type')['fuel_consumption'].mean().sort_values()
            avg_fuel.plot(kind='barh', ax=ax2)
            ax2.set_title("Avg Fuel by Ship Type")
            st.pyplot(fig2)

        with g2:
            fig3, ax3 = plt.subplots()
            monthly = df.groupby('month')['co2_emissions'].sum()
            monthly.plot(kind='line', marker='o', ax=ax3)
            ax3.set_title("Monthly CO₂ Emissions")
            st.pyplot(fig3)

            fig4, ax4 = plt.subplots()
            sns.countplot(data=df, x='cii_grade', order=list("ABCDE"), ax=ax4)
            ax4.set_title("CII Grade Distribution")
            st.pyplot(fig4)

    else:
        st.warning("Visualization data not available.")
