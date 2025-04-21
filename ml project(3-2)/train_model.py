# ✅ Training code
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load your dataset
df = pd.DataFrame({
    "pm2_5": np.random.randint(10, 200, 100),
    "pm10": np.random.randint(20, 300, 100),
    "no2": np.random.randint(5, 100, 100),
    "co": np.random.uniform(0.1, 2.0, 100),
    "so2": np.random.randint(2, 50, 100),
    "o3": np.random.randint(10, 120, 100),
    "future_pm2_5": np.random.randint(10, 200, 100)
})

# ✅ Feature columns should match prediction time
features = ['pm2_5', 'pm10', 'no2', 'co', 'so2', 'o3']
X = df[features]
y = df['future_pm2_5']  # or whatever your target label is

# ✅ Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train models
rf_model = RandomForestRegressor()
rf_model.fit(X_scaled, y)

lr_model = LinearRegression()
lr_model.fit(X_scaled, y)

# ✅ Save models and scaler
joblib.dump(rf_model, "models/rf_model.pkl")
joblib.dump(lr_model, "models/lr_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
