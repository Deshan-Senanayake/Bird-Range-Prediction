import joblib
import pandas as pd
import numpy as np
import re
import datetime
from difflib import get_close_matches
import joblib
import requests
import io


url2 = "https://raw.githubusercontent.com/Deshan-Senanayake/Bird-Range-Prediction/main/Migration%20model/models/location_prediction_model.pkl"


# Download the file
response2 = requests.get(url2)
response2.raise_for_status()  # Ensure we notice bad responses

# Load the model from memory
model_data2 = joblib.load(io.BytesIO(response2.content))

location_model = model_data2['location_model']
selected_features2 = model_data2['selected_features']
label_encoders2 = model_data2['label_encoders']

# ✅ Predefined Latitude & Longitude Values (Expand as Needed)
predefined_locations = [
    {"LATITUDE": 6.0463438, "LONGITUDE": 80.8541554},
    {"LATITUDE": 6.188598, "LONGITUDE": 81.2200356},
    {"LATITUDE": 6.1963995, "LONGITUDE": 81.2109113},
    {"LATITUDE": 6.1930548, "LONGITUDE": 81.2218203},
    {"LATITUDE": 6.0906125, "LONGITUDE": 80.9354124},
    {"LATITUDE": 6.188598, "LONGITUDE": 81.2200356},
    {"LATITUDE": 6.1930548, "LONGITUDE": 81.2218203}
]

# ✅ Bird Name Corrections (Handles NLP Variations)
bird_aliases = {
    "blue tailed bird": "Blue-tailed Bee-eater",
    "blue bird": "Blue-tailed Bee-eater",
    "bee eater": "Blue-tailed Bee-eater",
    "red bird": "Red-vented Bulbul",
    "bulbul": "Red-vented Bulbul",
    "white bird": "White-throated Kingfisher",
    "kingfisher": "White-throated Kingfisher"
}

valid_bird_names = ["Blue-tailed Bee-eater", "Red-vented Bulbul", "White-throated Kingfisher"]

# ✅ Function: Correct Bird Name with NLP
def correct_bird_name(name):
    name = name.lower()
    if name in bird_aliases:
        return bird_aliases[name]
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

# ✅ Function: Convert Time of Day to Hour Range
def time_of_day_to_hour(time_str):
    time_ranges = {
        "morning": (6, 10), "afternoon": (11, 15),
        "evening": (16, 19), "night": (20, 23)
    }
    return time_ranges.get(time_str.lower(), None)


# ✅ Bird Name Aliases
bird_aliases = {
    "blue tailed bird": "Blue-tailed Bee-eater",
    "blue bird": "Blue-tailed Bee-eater",
    "bee eater": "Blue-tailed Bee-eater",
    "red bird": "Red-vented Bulbul",
    "bulbul": "Red-vented Bulbul",
    "white bird": "White-throated Kingfisher",
    "kingfisher": "White-throated Kingfisher"
}

def extract_query_features_location(query):
    query = query.lower()
    
    # Extract Year
    year_match = re.search(r'\b(20[0-9]{2})\b', query)
    year = int(year_match.group()) if year_match else None

    # Extract Month
    months_map = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }
    month_match = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', query)
    month = months_map.get(month_match.group()) if month_match else None

    # Extract Day of Week
    day_name_match = re.search(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', query)
    day_of_week = day_name_to_int(day_name_match.group()) if day_name_match else None

    # Extract Time (Hour)
    time_match = re.search(r'\b([0-9]{1,2}):?([0-9]{2})?\s?(am|pm)?\b', query)
    if time_match:
        hour = int(time_match.group(1))
        if time_match.group(3) == "pm" and hour < 12:
            hour += 12
    else:
        time_match = re.search(r'\b(morning|afternoon|evening|night)\b', query)
        hour_range = time_of_day_to_hour(time_match.group()) if time_match else None
        hour = hour_range[0] if hour_range else None

    # Extract Bird Name (Allowing Both Alias & Actual Name)
    bird_name_match = re.search(r'\b(?:' + '|'.join([b.lower().replace("-", ".*") for b in valid_bird_names]) + r')\b', query)
    bird_name = bird_name_match.group() if bird_name_match else None

    # ✅ First, Check for Bird Alias Matches
    for alias, correct_name in bird_aliases.items():
        if alias in query:
            bird_name = correct_name  # Use mapped alias name
            break  # Stop after the first match

    # ✅ If No Alias Match, Ensure Actual Bird Name is in Correct Format
    if bird_name in valid_bird_names:
        bird_name = next(b for b in valid_bird_names if b.lower() == bird_name.lower())

    # ✅ Handle Missing Values With Default Assignments
    missing_inputs = []

    if year is None:
        year = 2025
        print("⚠️ Year is necessary to run the model. Since you didn't input it, year is 2025 by default. If you want another year, please enter it:")
        new_year = input().strip()
        if new_year.isdigit():
            year = int(new_year)

    if month is None:
        month = datetime.datetime.now().month  # Get current month
        print(f"⚠️ Month is necessary to run. Since you didn't input it, month is {month} by default. If you want another month, please enter it:")
        new_month = input().strip().lower()
        if new_month in months_map:
            month = months_map[new_month]

    if day_of_week is None:
        day_of_week = datetime.datetime.today().weekday()  # Get current day of the week
        print(f"⚠️ A day in the week is necessary to run. Since you didn't input it, day of week is {list(day_name_to_int.keys())[day_of_week].capitalize()} by default. If you want another day, please enter it (Monday-Sunday):")
        new_day = input().strip().lower()
        if new_day in day_name_to_int.keys():
            day_of_week = day_name_to_int(new_day)

    if hour is None:
        hour = datetime.datetime.now().hour  # Get current hour
        print(f"⚠️ Hour or a time period is necessary to run. Since you didn't input it, hour is {hour} by default. If you want another hour or a time period (morning, day, afternoon, evening), please enter it:")
        new_hour = input().strip().lower()
        if new_hour.isdigit():
            hour = int(new_hour)
        elif new_hour in time_of_day_to_hour.keys():
            hour = time_of_day_to_hour[new_hour][0]

    if bird_name is None or bird_name == "Unknown Bird":
        print("⚠️ A bird species should be entered to run the models. Please select a bird species from the list below:")
        print("\n".join(valid_bird_names))
        bird_name = input("Enter the correct bird species: ").strip()
        bird_name = correct_bird_name(bird_name)

    return {
        "year": year,
        "month": month,
        "day_of_week": day_of_week,
        "hour": hour,
        "bird_name": bird_name
    }


# ✅ Function: Predict Best Locations for Birdwatching
def predict_best_locations(query):
    features = extract_query_features_location(query)

    if "error" in features:
        return features

    # ✅ Ensure bird name is correctly formatted (fixes case issue)
    features["bird_name"] = next((b for b in valid_bird_names if b.lower() == features["bird_name"].lower()), features["bird_name"])

    # ✅ Get all known bird names from the encoder
    known_bird_names = set(label_encoders2['COMMON NAME'].classes_)

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

    results = []
    for location in predefined_locations:
        input_data = pd.DataFrame([[features["year"], features["month"], features["day_of_week"],
                                     features["hour"], location["LATITUDE"], 1, location["LONGITUDE"],
                                     label_encoders2['COMMON NAME'].transform([features["bird_name"]])[0]]],
                                  columns=selected_features2)

        predicted_location_encoded = location_model.predict(input_data)[0]
        predicted_location = label_encoders2['LOCALITY'].inverse_transform([predicted_location_encoded])[0]

        results.append(predicted_location)

    # ✅ Remove duplicate locations and print results
    unique_locations = list(set(results))

    print("\n✅ Recommended Locations for Birdwatching:\n")
    for idx, loc in enumerate(unique_locations, start=1):
        print(f"📍 Location {idx}: {loc}")

    return unique_locations
