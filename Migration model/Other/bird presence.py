import joblib
import pandas as pd
import numpy as np
import re
from difflib import get_close_matches
from rapidfuzz import process
import datetime

import joblib
import requests
import io

# Raw GitHub URL (change "blob/main" to "raw/main")
url1 = "https://raw.githubusercontent.com/Deshan-Senanayake/Bird-Range-Prediction/main/Migration%20model/models/migration_prediction_model.pkl"

# Download the file
response1 = requests.get(url1)
response1.raise_for_status()  # Ensure we notice bad responses

# Load the model from memory
model_data1 = joblib.load(io.BytesIO(response1.content))

# Extract trained model and features
rf_model = model_data1['rf_final']
label_encoders1 = model_data1['label_encoders']
selected_features1 = model_data1['selected_features']


# ✅ Locality and Bird Name Handling
valid_localities = [
    "Buckingham Place Hotel Tangalle", "Bundala NP General", "Bundala National Park", 
    "Kalametiya", "Tissa Lake", "Yala National Park General", "Debarawewa Lake"
]

valid_localities_lower = {b.lower(): b for b in valid_localities}

valid_bird_names = ["Blue-tailed Bee-eater", "Red-vented Bulbul", "White-throated Kingfisher"]

bird_aliases = {
    "blue tailed bird": "Blue-tailed Bee-eater",
    "blue bird": "Blue-tailed Bee-eater",
    "bee eater": "Blue-tailed Bee-eater",
    "red bird": "Red-vented Bulbul",
    "bulbul": "Red-vented Bulbul",
    "white bird": "White-throated Kingfisher",
    "kingfisher": "White-throated Kingfisher"
}

# ✅ Function: Handle Bird Name Variations
valid_bird_names_lower = {b.lower(): b for b in valid_bird_names}  # Precompute for fast lookup

def correct_bird_name(name):
    name = name.lower()
    if name in bird_aliases:
        return bird_aliases[name]
    return valid_bird_names_lower.get(name, "Unknown Bird")  # Direct lookup



def correct_locality(user_input):
    user_input = user_input.lower()

    # ✅ Check exact match
    if user_input in valid_localities_lower:
        return valid_localities_lower[user_input]

    # ✅ Check partial match
    for loc_lower, loc in valid_localities_lower.items():
        if user_input in loc_lower:
            return loc

    # ✅ Manual keyword-based corrections
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



# ✅ Function: Convert Day Name to Integer
def day_name_to_int(day_name):
    days_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, 
        "friday": 4, "saturday": 5, "sunday": 6
    }
    return days_map.get(day_name.lower(), "Invalid Day")  # Returns a string instead of None


# ✅ Function: Convert Time of Day to Hour Range
def time_of_day_to_hour(time_str):
    time_ranges = {
        "morning": [6, 10], "afternoon": [11, 15],
        "evening": [16, 19], "night": [20, 23]
    }
    return time_ranges.get(time_str.lower(), "Invalid Time")  # Avoids None issues


# ✅ Function: Extract Features from Query
import re

def extract_query_features_bird_presence(query):
    query = query.lower()

    # Extract Year (already provided by Rasa)
    year_match = re.search(r'\b(20[0-9]{2})\b', query)
    year = int(year_match.group()) if year_match else None  # No default, must be provided by Rasa

    # Extract Month (already provided by Rasa)
    months_map = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }
    month_match = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', query)
    month = months_map.get(month_match.group()) if month_match else None

    # Extract Day of the Week (already provided by Rasa)
    day_name_match = re.search(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', query)
    day_of_week = day_name_match.group() if day_name_match else None

    # Extract Time (already provided by Rasa)
    time_match = re.search(r'\b([0-9]{1,2}):?([0-9]{2})?\s?(am|pm)?\b', query)
    hour = None
    if time_match:
        hour = int(time_match.group(1))
        if time_match.group(3) == "pm" and hour < 12:
            hour += 12

    # Extract Locality (already provided by Rasa)
    locality_match = re.search(r'\b[a-zA-Z\s]+\b', query)
    locality = locality_match.group() if locality_match else None

    # Extract Bird Name (already provided by Rasa)
    bird_name_match = re.search(r'\b(?:' + '|'.join([b.lower().replace("-", ".*") for b in valid_bird_names]) + r')\b', query)
    bird_name = bird_name_match.group() if bird_name_match else None

    # Bird alias correction
    if bird_name:
        bird_name = bird_aliases.get(bird_name.lower(), bird_name)

    return {
        "year": year,
        "month": month,
        "day_of_week": day_of_week,
        "hour": hour,
        "locality": locality,
        "bird_name": bird_name
    }



import re

def generate_meaningful_query(result1, original_query):
    """
    Converts the structured prediction result into a meaningful sentence.
    Ensures the time of day matches the original query.
    """
    if "error" in result1:
        return result1["error"]

    presence_text = "is likely to be present" if result1["predicted_presence"] == 1 else "is unlikely to be present"
    
    # Ensure day_of_week is in proper format (Rasa now provides it as a string)
    days_map = {
        "monday": "Monday", "tuesday": "Tuesday", "wednesday": "Wednesday",
        "thursday": "Thursday", "friday": "Friday", "saturday": "Saturday", "sunday": "Sunday"
    }
    day_name = days_map.get(result1["features_used"]["day_of_week"].lower(), "Unknown Day")

    # Extract original time phrase from query (morning, afternoon, evening, night)
    time_of_day_match = re.search(r'\b(morning|afternoon|evening|night)\b', original_query.lower())
    if time_of_day_match:
        time_of_day = time_of_day_match.group()  # Get the exact phrase
    else:
        # Convert hour into a natural phrase if no explicit time was mentioned
        hour = result1["features_used"]["hour"]
        if 6 <= hour <= 10:
            time_of_day = "in the morning"
        elif 11 <= hour <= 15:
            time_of_day = "in the afternoon"
        elif 16 <= hour <= 19:
            time_of_day = "in the evening"
        elif 20 <= hour <= 23:
            time_of_day = "at night"
        else:
            time_of_day = f"at {hour}:00"  # Show exact time if out of range

    probability = result1["probability"] * 100  # Convert to percentage

    # Construct a meaningful sentence
    query_sentence = (
        f"The {result1['features_used']['bird_name']} {presence_text} "
        f"at {result1['features_used']['locality']} on {day_name}, {result1['features_used']['month']}/"
        f"{result1['features_used']['year']} {time_of_day}. (Confidence: {probability:.1f}%)"
    )

    return query_sentence



import pandas as pd
import re
from difflib import get_close_matches

def predict_bird_presence(features1):
    """
    Predicts bird presence based on extracted features.
    Assumes all required values are provided by Rasa before calling this function.
    """

    if not isinstance(features1, dict):
        return "Failed to extract features. Please check your query format."

    if "error" in features1:
        return features1["error"]

    if features1["locality"] == "Unknown Location":
        return "The locality entered is not recognized. Please provide a more specific location."

    features1["bird_name"] = next((b for b in valid_bird_names if b.lower() == features1["bird_name"].lower()), features1["bird_name"])

    known_bird_names = set(label_encoders1['COMMON NAME'].classes_)

    if features1["bird_name"] not in known_bird_names:
        closest_match = get_close_matches(features1["bird_name"], known_bird_names, n=1, cutoff=0.3)
        if closest_match:
            features1["bird_name"] = closest_match[0]
        else:
            return f"'{features1['bird_name']}' is not recognized. Please check the bird name."

    try:
        locality_encoded = label_encoders1['LOCALITY'].transform([features1["locality"]])[0]
        bird_name_encoded = label_encoders1['COMMON NAME'].transform([features1["bird_name"]])[0]

        input_data = pd.DataFrame([[features1["year"], features1["month"], features1["day_of_week"],
                                     features1["hour"], locality_encoded, bird_name_encoded]],
                                  columns=selected_features1)

        probability = rf_model.predict_proba(input_data)[:, 1][0]
        prediction = int(probability >= 0.5)

        result1 = {
            "predicted_presence": prediction,
            "probability": round(probability, 3),
            "features_used": features1
        }

        return generate_meaningful_query(result1, "User query")  # ✅ Return plain text for Rasa

    except Exception as e:
        return f"Prediction error: {str(e)}"
