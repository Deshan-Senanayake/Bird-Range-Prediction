import joblib
import pandas as pd
import numpy as np

# Load All Models and Encoders
migration_model = joblib.load(r'C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\migration_prediction_model.pkl')
location_model = joblib.load(r'C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\location_prediction_model.pkl')
time_model = joblib.load(r'C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\time_prediction_model.pkl')

rf_final = migration_model['rf_final']
location_rf = location_model['location_model']
month_model = time_model['month_model']
hour_model = time_model['hour_model']

# Load Label Encoders
le_species = migration_model['label_encoders']['COMMON NAME']
le_locality = migration_model['label_encoders']['LOCALITY']

# Load Dataset for Background Processing (Used for Location Prediction)
df_migration = pd.read_csv(r'C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\data\migration_data.csv')

# Helper Function: Convert Categorical Inputs to Encoded Values
def encode_inputs(species, locality):
    species_encoded = le_species.transform([species])[0] if species in le_species.classes_ else None
    locality_encoded = le_locality.transform([locality])[0] if locality in le_locality.classes_ else None
    return species_encoded, locality_encoded

# Helper Function: Convert Day Names to Integer (Monday = 0, Sunday = 6)
def convert_day_of_week(day_name):
    days = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    return days.get(day_name, None)

# Helper Function: Convert Time of Day to Hour Range
def convert_time_of_day(time_period):
    time_ranges = {
        'morning': (6, 10),
        'afternoon': (11, 15),
        'evening': (16, 19),
        'night': (20, 23)
    }
    return time_ranges.get(time_period.lower(), None)

# Model A - Presence Prediction Function
def predict_migration_presence(species, year, month, day_of_week, hour, locality):
    species_encoded, locality_encoded = encode_inputs(species, locality)
    
    if species_encoded is None or locality_encoded is None:
        return "Invalid species or locality. Please enter valid values."
    
    # Create Input DataFrame
    input_data = pd.DataFrame([[year, month, day_of_week, hour, locality_encoded, species_encoded]],
                              columns=['Year', 'Month', 'Day_of_Week', 'Hour', 'LOCALITY_ENCODED', 'COMMON NAME_ENCODED'])
    
    # Predict Probability of Presence
    prob = rf_final.predict_proba(input_data)[:, 1][0]
    
    # Convert to Yes/No
    presence = "Yes" if prob >= 0.5 else "No"
    return f"The likelihood of seeing {species} at {locality} on {month}/{day_of_week}/{year} at {hour}:00 is {presence} ({prob:.2%} confidence)."

# Model B - Location Prediction Function
def predict_bird_location(species, year, month, day_of_week, time_period):
    species_encoded, _ = encode_inputs(species, None)

    if species_encoded is None:
        return "Invalid species. Please enter a valid bird species."

    # Retrieve 20 Positive Predictions (Presence = Yes)
    positive_samples = df_migration[df_migration['OBSERVATION'] == 1].sample(n=20, random_state=42)
    
    # Extract Latitude, Longitude, and Localities
    latitudes = positive_samples['LATITUDE'].tolist()
    longitudes = positive_samples['LONGITUDE'].tolist()
    localities = positive_samples['LOCALITY'].tolist()

    # Convert Time of Day to Hour Range
    time_range = convert_time_of_day(time_period)
    if time_range is None:
        return "Invalid time period. Choose from morning, afternoon, evening, or night."
    
    start_hour, end_hour = time_range

    # Create Input DataFrame
    input_data = pd.DataFrame([[year, month, day_of_week, start_hour, lat, lon, species_encoded]
                                for lat, lon in zip(latitudes, longitudes)],
                                columns=['Year', 'Month', 'Day_of_Week', 'Hour', 'LATITUDE', 'LONGITUDE', 'COMMON NAME_ENCODED'])

    # Predict Best Locations
    probabilities = location_rf.predict_proba(input_data)[:, 1]
    
    # Attach Probabilities to Locations
    location_predictions = sorted(zip(localities, probabilities), key=lambda x: x[1], reverse=True)[:10]

    # Convert to User-Friendly Output
    response = f"Best locations to see {species} at {time_period}:\n"
    for loc, prob in location_predictions:
        response += f"- **{loc}** ({prob:.1%} probability)\n"

    return response

# Model C - Best Time Prediction Function
def predict_best_time(species, locality, season, time_period):
    species_encoded, locality_encoded = encode_inputs(species, locality)

    if species_encoded is None or locality_encoded is None:
        return "Invalid species or locality. Please enter valid values."

    # Initialize Default Values
    features = {
        'OBSERVATION': 1,  # Default to "Yes" (bird present)
        'Is_Summer': 0, 'Is_Winter': 0, 'Is_Spring': 0, 'Is_Autumn': 0,
        'Is_Morning': 0, 'Is_Afternoon': 0, 'Is_Evening': 0, 'Is_Night': 0
    }

    # Activate Season and Time
    if season in features:
        features[season] = 1
    if time_period in features:
        features[time_period] = 1

    # Create Input DataFrame
    input_data = pd.DataFrame([[1, species_encoded, locality_encoded] + list(features.values())],
                              columns=['OBSERVATION', 'COMMON NAME_ENCODED', 'LOCALITY_ENCODED'] + list(features.keys()))

    # Predict Best Month and Hour
    best_month = month_model.predict(input_data)[0]
    best_hour = hour_model.predict(input_data)[0]

    return f"The best time to see {species} at {locality} is in **Month {best_month}**, around **{best_hour}:00**."

# Main Script: User Input Handling
print("Choose an option:\n1. Check Bird Presence\n2. Find Best Locations\n3. Find Best Times")
choice = int(input("Enter your choice (1-3): "))

if choice == 1:
    print(predict_migration_presence("Blue-tailed Bee-eater", 2025, 3, 2, 7, "Bundala NP General"))
elif choice == 2:
    print(predict_bird_location("Blue-tailed Bee-eater", 2025, 3, 2, "morning"))
elif choice == 3:
    print(predict_best_time("Blue-tailed Bee-eater", "Bundala NP General", "Is_Summer", "Is_Morning"))
