import pandas as pd
from dataprep.clean import clean
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("ship_fuel_efficiency.csv")  # Replace with correct path if needed
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Automated cleaning using DataPrep
for col in ['ship_type', 'route_id', 'month', 'fuel_type', 'weather_conditions']:
    df = clean(df, col)

# Drop empty columns and rows with missing values
df = df.dropna(axis=1, how='all')
df = df.dropna()

# Drop non-essential columns
drop_cols = ['ship_id', 'co2_emissions', 'engine_efficiency']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Define target
target = 'fuel_consumption' if 'fuel_consumption' in df.columns else 'fuel_consumption_(mt)'

# Encode categorical columns
categorical = ['ship_type', 'fuel_type', 'route_id', 'month', 'weather_conditions']
encoders = {}
for col in categorical:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Define features and labels
X = df.drop(columns=[target])
y = df[target]

# Scale numeric columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Save model, scaler, and encoders
joblib.dump({
    'model': model,
    'scaler': scaler,
    'encoders': encoders,
    'features': list(X.columns)
}, 'fuel_predictor.pkl')
