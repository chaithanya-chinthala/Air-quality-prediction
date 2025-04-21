import pandas as pd
import pickle

# Load models and scaler
with open("models/rf_model.pkl", "rb") as rf_file:
    rf_model = pickle.load(rf_file)

with open("models/lr_model.pkl", "rb") as lr_file:
    lr_model = pickle.load(lr_file)

with open("models/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Example new data
new_data = pd.DataFrame({
    "pm2_5": [40],
    "pm10": [60],
    "no2": [30],
    "co": [1.5],
    "so2": [20],
    "o3": [50]
})

# Scale new data
new_data_scaled = scaler.transform(new_data)

# Predict using Random Forest
rf_prediction = rf_model.predict(new_data_scaled)
print(f"🌲 Random Forest Predicted Future PM2.5: {rf_prediction[0]}")

# Predict using Linear Regression
lr_prediction = lr_model.predict(new_data_scaled)
print(f"📈 Linear Regression Predicted Future PM2.5: {lr_prediction[0]}")
