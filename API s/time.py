from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import re
import datetime
from difflib import get_close_matches
from rapidfuzz import process
import requests
import io
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load Model & Encoders from GitHub
MODEL_URL = "https://raw.githubusercontent.com/Deshan-Senanayake/Bird-Range-Prediction/main/Migration%20model/models/time_prediction_model.pkl"

def load_model():
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    return joblib.load(io.BytesIO(response.content))

model_data = load_model()
month_model = model_data['month_model']
hour_model = model_data['hour_model']
selected_features = model_data['selected_features']
label_encoders = model_data['label_encoders']

# ✅ Define Valid Localities and Bird Names
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

season_aliases = {"summer": "Is_Summer", 
                  "winter": "Is_Winter", 
                  "spring": "Is_Spring", 
                  "autumn": "Is_Autumn"}

time_period_aliases = {"morning": "Is_Morning", 
                       "afternoon": "Is_Afternoon", 
                       "evening": "Is_Evening", 
                       "night": "Is_Night"}

# ✅ Helper Function: Correct Bird Name
def correct_bird_name(name):
    name = name.lower()
    if name in bird_aliases:
        return bird_aliases[name]
    matches = get_close_matches(name, [b.lower() for b in valid_bird_names], n=1, cutoff=0.3)
    return next((b for b in valid_bird_names if b.lower() == matches[0]), "Unknown Bird")

# ✅ Helper Function: Convert Day Name to Integer
def day_name_to_int(day_name):
    days_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }
    return days_map.get(day_name.lower(), None)

# ✅ Helper Function: Correct Locality
def correct_locality(user_input):
    user_input = user_input.lower()
    for loc in valid_localities:
        if user_input == loc.lower():
            return loc
        if user_input in loc.lower():
            return loc  
        
    manual_mappings = {
        "bundala": "Bundala NP General",
        "yala": "Yala National Park General",
        "tissa": "Tissa Lake",
        "debara": "Debarawewa Lake",
        "kalametiya": "Kalametiya Bird Sanctuary"
    }
    return manual_mappings.get(user_input, "Unknown Location")


def parse_approximate_date(expression):
    today = datetime.date.today()
    date_mappings = {
        "tomorrow": today + datetime.timedelta(days=1),
        "day after tomorrow": today + datetime.timedelta(days=2),
        "next week": today + datetime.timedelta(weeks=1)
        
    }
    if expression in date_mappings:
        return date_mappings[expression]
    match = re.search(r"in (\d+) days", expression)
    if match:
        return today + datetime.timedelta(days=int(match.group(1)))
    return None  

def get_current_season():
    """Returns the current season based on the current month."""
    month = datetime.datetime.today().month
    if month in [12, 1, 2]:  # December, January, February
        return "Is_Winter"
    elif month in [3, 4, 5]:  # March, April, May
        return "Is_Spring"
    elif month in [6, 7, 8]:  # June, July, August
        return "Is_Summer"
    else:  # September, October, November
        return "Is_Autumn"

# ✅ Extract Features from Query
def extract_query_features_time(query):
    query = query.lower()
    today = datetime.date.today()
    
    # ✅ Extract Year (Defaults to Current Year)
    year_match = re.search(r'\b(20[0-9]{2})\b', query)
    year = int(year_match.group()) if year_match else today.year
    

    
    
    day_name_match = re.search(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', query)
    day_of_week = day_name_to_int(day_name_match.group()) if day_name_match else datetime.datetime.today().weekday()
    
    day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day_of_week]

    

    locality = None
    for loc in valid_localities:
        if loc.lower() in query:
            locality = loc
            break  # Stop at first match
    
    # ✅ Handle Locality Aliases (Bundala -> Bundala NP General)
    if not locality:
        for alias, correct_loc in {
            "bundala": "Bundala NP General",
            "yala": "Yala National Park General",
            "tissa": "Tissa Lake",
            "debara": "Debarawewa Lake",
            "kalametiya": "Kalametiya Bird Sanctuary"
        }.items():
            if alias in query:
                locality = correct_loc
                break
            
    bird_name = None
    for bird in valid_bird_names:
        if bird.lower() in query:
            bird_name = bird
            break
    
    # ✅ Handle Bird Aliases (blue bird -> Blue-tailed Bee-eater)
    if not bird_name:
        for alias, correct_name in bird_aliases.items():
            if alias in query:
                bird_name = correct_name
                break

    season_flags = {season: 0 for season in season_aliases.values()}
    found_season = False

    for season, flag in season_aliases.items():
        if season in query:
            season_flags[flag] = 1
            found_season = True

    if not found_season:  # ✅ If no season found, use the current season
        current_season_flag = get_current_season()
        season_flags[current_season_flag] = 1
    
    time_period_flags = {time: 0 for time in time_period_aliases.values()}

    for season, flag in season_aliases.items():
        if season in query:
            season_flags[flag] = 1

    for time, flag in time_period_aliases.items():
        if time in query:
            time_period_flags[flag] = 1

    return {
        "year": year,
        "day_of_week": day_of_week,
        "locality": locality,
        "bird_name": bird_name,
        "day_name": day_name,
        **season_flags,
        **time_period_flags
    }

# ✅ API Endpoint for Rasa Chatbot
@app.route('/predict_best_time', methods=['POST'])
def predict_best_time():
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        logger.info(f"🔍 Received Query: {query}")

        if not query:
            return jsonify({"error": "No query provided"}), 400

        features = extract_query_features_time(query)

        # ✅ Ensure Locality and Bird Name Are Not Missing Before Encoding
        if features["locality"] == "Unknown Location" or features["locality"] is None:
            return jsonify({
                "message": "The query you entered didn't contain a location. Please select one.",
                "valid_localities": valid_localities,
                "location_aliases": [
                    "You can also use 'Bundala' instead of 'Bundala NP General'.",
                    "You can use 'Yala' instead of 'Yala National Park General'.",
                    "You can use 'Tissa' instead of 'Tissa Lake'."
                ]
            }), 400  # ✅ Make sure we return and STOP execution

        if features["bird_name"] == "Unknown Bird" or features["bird_name"] is None:
            return jsonify({
                "message": "The query you entered didn't contain a bird species. Please select one and re-enter the query.",
                "valid_bird_names": valid_bird_names
            }), 400  # ✅ Ensure we return and STOP execution

        # ✅ Encode Locality & Bird Name
        try:
            if features["locality"] not in valid_localities:
                raise ValueError(f"Invalid locality: {features['locality']}")

            if features["bird_name"] not in valid_bird_names:
                raise ValueError(f"Invalid bird name: {features['bird_name']}")

            locality_encoded = label_encoders['LOCALITY'].transform([features["locality"]])[0]
            bird_name_encoded = label_encoders['COMMON NAME'].transform([features["bird_name"]])[0]

        except ValueError as e:
            logger.error(f"Encoding Error: {e}")
            return jsonify({"error": f"Invalid input detected: {str(e)}"}), 400  # ✅ Return proper error message



        input_data = pd.DataFrame([[1, features["year"], features["day_of_week"],
                                    locality_encoded, bird_name_encoded,
                                    features["Is_Summer"], features["Is_Winter"], features["Is_Spring"], features["Is_Autumn"],
                                    features["Is_Morning"], features["Is_Afternoon"], features["Is_Evening"], features["Is_Night"]]],
                                columns=selected_features)

        predicted_month = int(round(month_model.predict(input_data)[0]))
        predicted_hour = int(round(hour_model.predict(input_data)[0]))

        months_map = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
                      7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
        month_name = months_map.get(predicted_month, f"Unknown ({predicted_month})")

        am_pm = "a.m." if predicted_hour < 12 else "p.m."
        formatted_hour = predicted_hour if predicted_hour <= 12 else predicted_hour - 12
        if formatted_hour == 0:
            formatted_hour = 12

        response = {
            "Response": (
                f"The {features['bird_name']} can be seen "
                f"at {features['locality']} on a {features['day_name']}, "
                f"at {formatted_hour}:00 {am_pm} "
                f"in {month_name}."
            )
        }

        return jsonify(response), 200


    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}", "status": "failure"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
