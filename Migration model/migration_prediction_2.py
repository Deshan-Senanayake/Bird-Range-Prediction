import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# ✅ Load Trained Models & Encoders
presence_model = joblib.load("models/migration_prediction_model.pkl")["model"]
time_models = joblib.load("models/time_prediction_model.pkl")
location_model = joblib.load("models/location_prediction_model.pkl")

county_encoder = time_models["county_encoder"]
locality_encoder = time_models["locality_encoder"]

month_model = time_models["month_model"]
day_model = time_models["day_model"]
hour_model = time_models["hour_model"]

# ✅ Function to Predict Bird Presence
def predict_bird_presence(species, date, time, county, locality):
    """
    Predicts whether a given bird species will be observed at a future date, time, and location.
    
    Args:
        species (str): Bird species name.
        date (str): Future date in 'YYYY-MM-DD' format.
        time (str): Future time in 'HH:MM' format.
        county (str): Name of the county.
        locality (str): Name of the locality.
    
    Returns:
        str: Prediction result.
    """
    # ✅ Convert Date & Time to Model Format
    datetime_obj = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
    year, month, day, hour, day_of_week = (
        datetime_obj.year, datetime_obj.month, datetime_obj.day, datetime_obj.hour, datetime_obj.weekday()
    )

    # ✅ Encode County & Locality
    county_encoded = county_encoder.transform([county])[0] if county in county_encoder.classes_ else -1
    locality_encoded = locality_encoder.transform([locality])[0] if locality in locality_encoder.classes_ else -1

    # ✅ Predict Time (Month, Day, Hour) if user didn’t specify
    predicted_month = month_model.predict([[county_encoded, locality_encoded]])[0]
    predicted_day = day_model.predict([[county_encoded, locality_encoded]])[0]
    predicted_hour = hour_model.predict([[county_encoded, locality_encoded]])[0]

    # ✅ Use User Input If Provided
    final_month = int(predicted_month) if month is None else month
    final_day = int(predicted_day) if day is None else day
    final_hour = int(predicted_hour) if hour is None else hour

    # ✅ Predict Location if user didn’t specify
    location_prediction = location_model.predict([[year, final_month, final_day, day_of_week, final_hour]])
    predicted_locality = locality_encoder.inverse_transform([int(location_prediction[0])])[0]

    # ✅ Prepare Feature Vector for Bird Presence Prediction
    features = np.array([[year, final_month, final_day, day_of_week, final_hour, county_encoded, locality_encoded]])

    # ✅ One-Hot Encode Species
    species_column = f"COMMON NAME_{species}"
    species_vector = np.zeros((1, len(presence_model.feature_names_in_)))
    
    if species_column in presence_model.feature_names_in_:
        species_index = np.where(presence_model.feature_names_in_ == species_column)[0][0]
        species_vector[0, species_index] = 1
    
    # ✅ Combine Features
    final_input = np.hstack([features, species_vector])

    # ✅ Predict Bird Presence
    presence_prob = presence_model.predict_proba(final_input)[0][1]  # Probability of presence

    # ✅ Convert to Human-Readable Format
    presence_likelihood = round(presence_prob * 100, 2)

    return f"The {species} has a **{presence_likelihood}% chance** of being observed in **{predicted_locality}** on **{final_day}/{final_month}/{year} at {final_hour}:00**."

# ✅ Example Usage
species_query = "Blue-tailed Bee-eater"
date_query = "2025-03-12"
time_query = "07:00"
county_query = "Beliatta"
locality_query = "Muruthawela Lake"

result = predict_bird_presence(species_query, date_query, time_query, county_query, locality_query)
print(result)
