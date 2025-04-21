from flask import Flask, render_template, request
import requests
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

API_KEY = "934b67a8ab09d6e5cc07a36f29395c93"  # Replace with your actual API key

with open("models/rf_model.pkl", "rb") as rf_file:
    rf_model = joblib.load(rf_file)
    print("Random Forest model loaded successfully.")

with open("models/lr_model.pkl", "rb") as lr_file:
    lr_model = joblib.load(lr_file)
    print("Linear Regression model loaded successfully.")

with open("models/scaler.pkl", "rb") as scaler_file:
    scaler = joblib.load(scaler_file)
    print("Scaler loaded successfully.")

# Function to fetch air quality data
def get_air_quality(city):
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
    response = requests.get(weather_url)
    data = response.json()

    print(f"API Response: {data}")  # Debugging the API response
    
    if response.status_code == 200 and "coord" in data:
        lat = data["coord"]["lat"]
        lon = data["coord"]["lon"]

        pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        pollution_response = requests.get(pollution_url)
        pollution_data = pollution_response.json()

        print(f"Pollution Data: {pollution_data}")

        if "list" in pollution_data and pollution_data["list"]:
            pollutants = pollution_data["list"][0]["components"]
            if all(value == 0 for value in pollutants.values()):
                print("⚠️ Warning: All pollutant values are zero.")
                return None, None, None
            return pollutants, lat, lon

    return None, None, None  # Add this to handle bad API responses


def calculate_aqi_pm25(concentration):
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ]

    for bp_low, bp_high, aqi_low, aqi_high in breakpoints:
        if bp_low <= concentration <= bp_high:
            aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (concentration - bp_low) + aqi_low
            return round(aqi)
    return 500  # If value exceeds the last range


# Function to classify AQI level and suggestion
def classify_air_quality(aqi):
    if aqi <= 50:
        return "Good", "Air quality is satisfactory."
    elif aqi <= 100:
        return "Moderate", "Air quality is acceptable but may be a concern for sensitive groups."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "People with respiratory issues should limit outdoor activity."
    elif aqi <= 200:
        return "Unhealthy", "Everyone may begin to experience adverse health effects."
    elif aqi <= 300:
        return "Very Unhealthy", "Health alert: everyone should avoid outdoor exposure."
    else:
        return "Hazardous", "Serious health effects. Avoid all outdoor activity."

@app.route("/", methods=["GET", "POST"])
def index():
    air_quality_data = None
    error = None
    category = None
    suggestion = None
    city = None
    predicted_aqi = None
    current_aqi = None
    current_category = None
    current_suggestion = None
    predicted_category = None
    predicted_suggestion = None

    if request.method == "POST":
        city = request.form["city"]
        air_quality_data, lat, lon = get_air_quality(city)

        if air_quality_data:
            pm2_5 = air_quality_data.get("pm2_5", 0)
            pm10 = air_quality_data.get("pm10", 0)
            no2 = air_quality_data.get("no2", 0)
            co = air_quality_data.get("co", 0)
            so2 = air_quality_data.get("so2", 0)
            o3 = air_quality_data.get("o3", 0)

            current_aqi = calculate_aqi_pm25(pm2_5)
            current_category, current_suggestion = classify_air_quality(current_aqi)

            input_features = np.array([[pm2_5, pm10, no2, co, so2, o3]])
            input_scaled = scaler.transform(input_features)

            # Predict using models only if current PM2.5 is reasonably high
            if pm2_5 >= 25:
                future_pm2_5_rf = rf_model.predict(input_scaled)[0]
                future_pm2_5_lr = lr_model.predict(input_scaled)[0]
                #predicted_pm2_5 = (future_pm2_5_rf + future_pm2_5_lr) / 2

                # Clip predicted PM2.5 to realistic range
                predicted_pm2_5 = (0.7 * future_pm2_5_rf + 0.3 * future_pm2_5_lr)

            else:
                 predicted_pm2_5 = pm2_5

            predicted_aqi = calculate_aqi_pm25(predicted_pm2_5)
            predicted_category, predicted_suggestion = classify_air_quality(predicted_aqi)

            print("City:", city)
            print("Input Features:", input_features)
            print("Scaled Input:", input_scaled)
            print("Predicted PM2.5 RF:", future_pm2_5_rf if pm2_5 >= 25 else "Skipped")
            print("Predicted PM2.5 LR:", future_pm2_5_lr if pm2_5 >= 25 else "Skipped")
            print("Final Predicted PM2.5:", predicted_pm2_5)
            print("Predicted AQI:", predicted_aqi)


        else:
            error = "City not found or air quality data is unavailable."

    return render_template(
        "index.html",
        city=city,
        current_aqi=current_aqi,
        current_category=current_category,
        current_suggestion=current_suggestion,
        aqi=predicted_aqi,
        predicted_category=predicted_category,
        predicted_suggestion=predicted_suggestion,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)

