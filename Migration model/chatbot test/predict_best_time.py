import joblib
import pandas as pd
import numpy as np
import re
import datetime
from difflib import get_close_matches
from rapidfuzz import process

import requests
import io


url3 = "https://raw.githubusercontent.com/Deshan-Senanayake/Bird-Range-Prediction/main/Migration%20model/models/time_prediction_model.pkl"


# Download the file
response3 = requests.get(url3)
response3.raise_for_status()  # Ensure we notice bad responses

# Load the model from memory
model_data3 = joblib.load(io.BytesIO(response3.content))

month_model = model_data3['month_model']
hour_model = model_data3['hour_model']
selected_features = model_data3['selected_features']
label_encoders = model_data3['label_encoders']

valid_localities = [
    "Buckingham Place Hotel Tangalle", "Bundala NP General", "Bundala National Park", 
    "Kalametiya", "Tissa Lake", "Yala National Park General", "Debarawewa Lake"
]

valid_bird_names = ["Blue-tailed Bee-eater", "Red-vented Bulbul", "White-throated Kingfisher"]

bird_aliases = {
    "blue tailed bird": "Blue-tailed Bee-eater",
    "bee eater": "Blue-tailed Bee-eater",
    "blue bird": "Blue-tailed Bee-eater",
    "red bird": "Red-vented Bulbul",
    "bulbul": "Red-vented Bulbul",
    "white bird": "White-throated Kingfisher",
    "kingfisher": "White-throated Kingfisher"
}

season_aliases = {
    "summer": "Is_Summer",
    "winter": "Is_Winter",
    "spring": "Is_Spring",
    "autumn": "Is_Autumn"
}

time_period_aliases = {
    "morning": "Is_Morning",
    "afternoon": "Is_Afternoon",
    "evening": "Is_Evening",
    "night": "Is_Night"
}

# ✅ Function: Handle Bird Name Variations
def correct_bird_name(name):
    name = name.lower().strip()
    
    # ✅ Check in aliases first
    if name in bird_aliases:
        return bird_aliases[name]

    # ✅ Check for closest match in valid bird names
    matches = get_close_matches(name, [b.lower() for b in valid_bird_names], n=1, cutoff=0.3)
    if matches:
        return next(b for b in valid_bird_names if b.lower() == matches[0])

    return "Unknown Bird"

# ✅ Function: Convert Day Name to Integer
def day_name_to_int(day_name):
    days_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, 
        "friday": 4, "saturday": 5, "sunday": 6
    }
    return days_map.get(day_name.lower(), None)

# ✅ Function: Handle Locality Variations
def correct_locality(user_input):
    user_input = user_input.lower().strip()

    # ✅ Exact match
    for loc in valid_localities:
        if user_input == loc.lower():
            return loc

    # ✅ Partial match
    for loc in valid_localities:
        if user_input in loc.lower():
            return loc

    # ✅ Manual mappings
    manual_mappings = {
        "bundala": "Bundala NP General",
        "yala": "Yala National Park General",
        "tissa": "Tissa Lake",
        "debara": "Debarawewa Lake",
        "kalametiya": "Kalametiya Bird Sanctuary"
    }

    for keyword, mapped_location in manual_mappings.items():
        if keyword in user_input:
            return mapped_location

    return "Unknown Location"

# ✅ Function: Extract Features from Query
def extract_query_features_time(query):
    query = query.lower().strip()
    
    # Extract Year
    year_match = re.search(r'\b(20[0-9]{2})\b', query)
    year = int(year_match.group()) if year_match else 2025  # ✅ Default to 2025
    if not year_match:
        print("⚠️ Year is necessary to run the model. Since you didn't input it, year is 2025 by default. If you want another year, please enter it:")
        new_year = input().strip()
        if new_year.isdigit():
            year = int(new_year)

    # Extract Day of Week
    day_name_match = re.search(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', query)
    day_of_week = day_name_to_int(day_name_match.group()) if day_name_match else datetime.datetime.today().weekday()
    if not day_name_match:
        print(f"⚠️ A day in the week is necessary to run. Since you didn't input it, day of week is {list(day_name_to_int.keys())[day_of_week].capitalize()} by default. If you want another day, please enter it (Monday-Sunday):")
        new_day = input().strip().lower()
        if new_day in day_name_to_int.keys():
            day_of_week = day_name_to_int(new_day)

    # Extract Bird Name
    bird_name = None
    for alias, correct_name in bird_aliases.items():
        if alias in query:
            bird_name = correct_name
            break
    if not bird_name:
        bird_name_match = re.search(r'\b(?:' + '|'.join([b.lower().replace("-", ".*") for b in valid_bird_names]) + r')\b', query)
        bird_name = bird_name_match.group() if bird_name_match else "Unknown Bird"

    if bird_name == "Unknown Bird":
        print("⚠️ A bird species should be entered to run the models. Please select a bird species:")
        print("\n".join(valid_bird_names))
        bird_name = input("Enter the correct bird species: ").strip()
        bird_name = correct_bird_name(bird_name)

    # Extract Locality
    locality_match = re.search(r'\b[a-zA-Z\s]+\b', query)
    locality = correct_locality(locality_match.group()) if locality_match else "Unknown Location"
    if locality == "Unknown Location":
        print("⚠️ A location should be entered to run the models. Please select a location:")
        print("\n".join(valid_localities))
        locality = input("Enter the correct locality: ").strip()
        locality = correct_locality(locality)

    # Extract Season & Time Period
    season_flags = {season: 0 for season in season_aliases.values()}
    time_period_flags = {time: 0 for time in time_period_aliases.values()}

    for season, flag in season_aliases.items():
        if season in query:
            season_flags[flag] = 1

    for time, flag in time_period_aliases.items():
        if time in query:
            time_period_flags[flag] = 1

    if sum(season_flags.values()) == 0:
        print("⚠️ The season is necessary. Please select one:\nSummer\nWinter\nSpring\nAutumn")
        user_season = input("Enter the season: ").strip().lower()
        if user_season in season_aliases:
            season_flags[season_aliases[user_season]] = 1

    if sum(time_period_flags.values()) == 0:
        print("⚠️ The time period is necessary. Please select one:\nMorning\nAfternoon\nEvening\nNight")
        user_time_period = input("Enter the time period: ").strip().lower()
        if user_time_period in time_period_aliases:
            time_period_flags[time_period_aliases[user_time_period]] = 1

    return {
        "year": year,
        "day_of_week": day_of_week,
        "locality": locality,
        "bird_name": bird_name,
        **season_flags,
        **time_period_flags
    }

# ✅ Function: Predict Best Time for Birdwatching (Same as before, no changes needed)


# ✅ Function: Predict Best Time for Birdwatching
# ✅ Function: Predict Best Time for Birdwatching
def predict_best_time(query):
    features = extract_query_features_time(query)

    if "error" in features:
        return features

    # ✅ Ensure bird name is correctly formatted (fixes case issue)
    features["bird_name"] = next((b for b in valid_bird_names if b.lower() == features["bird_name"].lower()), features["bird_name"])

    # ✅ Get all known bird names from the encoder
    known_bird_names = set(label_encoders['COMMON NAME'].classes_)

    # ✅ Check if the bird name exists in known labels
    if features["bird_name"] not in known_bird_names:
        print(f"⚠️ Warning: '{features['bird_name']}' is not in the trained bird name labels.")
        # ✅ Find the closest valid bird name (fuzzy matching)
        closest_match = get_close_matches(features["bird_name"], known_bird_names, n=1, cutoff=0.3)
        if closest_match:
            print(f"✅ Using closest match: {closest_match[0]}")
            features["bird_name"] = closest_match[0]
        else:
            return {"error": f"'{features['bird_name']}' is not recognized. Please check the bird name."}

    try:
        # ✅ Encode locality and bird name
        locality_encoded = label_encoders['LOCALITY'].transform([features["locality"]])[0]
        bird_name_encoded = label_encoders['COMMON NAME'].transform([features["bird_name"]])[0]

        # ✅ Prepare Input Data
        input_data = pd.DataFrame([[1, features["year"], features["day_of_week"],
                                    locality_encoded, bird_name_encoded, 
                                    features["Is_Summer"], features["Is_Winter"], features["Is_Spring"], features["Is_Autumn"],
                                    features["Is_Morning"], features["Is_Afternoon"], features["Is_Evening"], features["Is_Night"]]],
                                columns=selected_features)

        # ✅ Predict Best Month & Hour
        predicted_month = int(round(month_model.predict(input_data)[0]))
        predicted_hour = int(round(hour_model.predict(input_data)[0]))

        # ✅ Convert Month Number to Month Name
        months_map = {
            1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
            7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
        }
        month_name = months_map.get(predicted_month, f"Unknown ({predicted_month})")

        # ✅ Convert Hour to AM/PM Format
        am_pm = "a.m." if predicted_hour < 12 else "p.m."
        formatted_hour = predicted_hour if predicted_hour <= 12 else predicted_hour - 12
        if formatted_hour == 0:
            formatted_hour = 12  # Convert 0-hour to 12 AM

        # ✅ Return Formatted Output
        return (
            f"✅ Best Time for Birdwatching:\n"
            f"📅 Month: {month_name} ({predicted_month})\n"
            f"⏰ Hour: {formatted_hour}:00 {am_pm}"
        )

    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}
