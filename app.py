import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🚢 Fuel Dashboard", layout="wide")
st.title("📊 Ship Fuel Analytics Dashboard")

# Load the historical CSV data
df = pd.read_csv("ship_fuel_efficiency.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df = df.dropna()

# Emission Factors (tons CO₂/ton fuel)
emission_factors = {'HFO': 3.114, 'MGO': 3.206, 'LNG': 2.750}

df['emission_factor'] = df['fuel_type'].map(emission_factors)
df['co2_emitted'] = df['fuel_consumption'] * df['emission_factor']

# Ask for cargo weight input to calculate EEOI & CII
with st.sidebar:
    cargo_weight = st.number_input("Enter average Cargo Weight (tons)", value=10000.0)

# Calculate EEOI and CII
df['eeoi'] = df['co2_emitted'] / (cargo_weight * df['distance'])
df['cii'] = (df['co2_emitted'] * 1_000_000) / (cargo_weight * df['distance'])

# Assign CII Grades
def get_cii_grade(cii):
    if cii < 5:
        return 'A'
    elif cii < 10:
        return 'B'
    elif cii < 15:
        return 'C'
    elif cii < 20:
        return 'D'
    else:
        return 'E'

df['cii_grade'] = df['cii'].apply(get_cii_grade)

# Section 1: Fuel vs Distance by Ship Type
st.subheader("🛢️ Fuel Consumption vs Distance")
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=df, x='distance', y='fuel_consumption', hue='ship_type', alpha=0.6, ax=ax1)
ax1.set_xlabel("Distance (NM)")
ax1.set_ylabel("Fuel Consumption (MT)")
ax1.set_title("Fuel Consumption by Distance Travelled")
st.pyplot(fig1)

# Section 2: Average Fuel by Ship Type
st.subheader("🚢 Average Fuel Consumption per Ship Type")
avg_fuel = df.groupby('ship_type')['fuel_consumption'].mean().sort_values(ascending=False)
st.bar_chart(avg_fuel)

# Section 3: Monthly CO₂ Emissions
st.subheader("📆 CO₂ Emission Trend by Month")
monthly = df.groupby('month')['co2_emitted'].mean()
st.line_chart(monthly)

# Section 4: CII Grade Distribution
st.subheader("🏷️ CII Grade Distribution")
grade_counts = df['cii_grade'].value_counts().sort_index()
st.bar_chart(grade_counts)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Data from ship_fuel_efficiency.csv")
