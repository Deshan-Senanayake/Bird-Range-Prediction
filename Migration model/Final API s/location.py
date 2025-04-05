from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import re
import datetime
from difflib import get_close_matches
import requests
import io
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load trained model from GitHub
# MODEL_URL = "https://raw.githubusercontent.com/Deshan-Senanayake/Bird-Range-Prediction/main/Migration%20model/models/location_prediction_model.pkl"
# response = requests.get(MODEL_URL)
# response.raise_for_status()
# model_data = joblib.load(io.BytesIO(response.content))

# location_model = model_data['location_model']
# selected_features = model_data['selected_features']
# label_encoders = model_data['label_encoders']


import requests
import joblib
import io
import os

# ✅ Model URL
MODEL_URL = "https://raw.githubusercontent.com/Deshan-Senanayake/Bird-Range-Prediction/main/Migration%20model/models/location_prediction_model.pkl"

# ✅ Define a local model path
MODEL_PATH = "location_prediction_model.pkl"

# ✅ Function: Download Large Model in Chunks
def download_model(url, save_path, chunk_size=1024 * 1024):  # 1MB chunks
    if os.path.exists(save_path):  # ✅ Skip download if file exists
        print(f"📁 Using cached model: {save_path}")
        return save_path

    print("📥 Downloading model. Please wait...")
    
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # ✅ Check for errors
        total_size = int(response.headers.get("content-length", 0))  # ✅ Get file size
        downloaded = 0
        
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    print(f"🔄 Downloaded: {downloaded / total_size:.2%}", end="\r")  # ✅ Show progress

    print("\n✅ Model downloaded successfully.")
    return save_path

# ✅ Download model if not cached
model_file_path = download_model(MODEL_URL, MODEL_PATH)

# ✅ Load Model
with open(model_file_path, "rb") as model_file:
    model_data = joblib.load(model_file)

location_model = model_data['location_model']
selected_features = model_data['selected_features']
label_encoders = model_data['label_encoders']

print("✅ Model loaded successfully!")


# Predefined Latitude & Longitude Values
predefined_locations = [
    {"LATITUDE": 6.0463438, "LONGITUDE": 80.8541554},
    {"LATITUDE": 6.188598, "LONGITUDE": 81.2200356},
    {"LATITUDE": 6.1963995, "LONGITUDE": 81.2109113},
    {"LATITUDE": 6.1930548, "LONGITUDE": 81.2218203},
    {"LATITUDE": 6.0906125, "LONGITUDE": 80.9354124},
    {"LATITUDE": 6.188598, "LONGITUDE": 81.2200356},
    {"LATITUDE": 6.1930548, "LONGITUDE": 81.2218203}
]

# Bird Name Handling
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

def correct_bird_name(name):
    name = name.lower()
    if name in bird_aliases:
        return bird_aliases[name]
    matches = get_close_matches(name, [b.lower() for b in valid_bird_names], n=1, cutoff=0.3)
    return next((b for b in valid_bird_names if b.lower() == matches[0]), "Unknown Bird")

def day_name_to_int(day_name):
    days_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }
    return days_map.get(day_name.lower(), None)

def time_of_day_to_hour(time_str):
    time_ranges = {"morning": (6, 10), "afternoon": (11, 15), "evening": (16, 19), "night": (20, 23)}
    return time_ranges.get(time_str.lower(), None)

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

# Extract query features
def extract_query_features(query):
    query = query.lower()
    today = datetime.date.today()
    
    # ✅ Extract Year (Defaults to Current Year)
    year_match = re.search(r'\b(20[0-9]{2})\b', query)
    year = int(year_match.group()) if year_match else today.year

    # ✅ Extract Month (Defaults to Current Month)
    months_map = {m: i+1 for i, m in enumerate([
        "january", "february", "march", "april", "may", "june", "july", 
        "august", "september", "october", "november", "december"
    ])}
    month_match = re.search(r'\b(' + '|'.join(months_map.keys()) + r')\b', query)
    month = months_map.get(month_match.group()) if month_match else today.month

    # ✅ Extract Day Name (If Provided)
    days_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }
    day_name_match = re.search(r"\b(" + "|".join(days_map.keys()) + r")\b", query)
    day_name = day_name_match.group().capitalize() if day_name_match else None

    # ✅ Extract Approximate Date (e.g., "tomorrow", "next week")
    approximate_date_match = re.search(r"(tomorrow|next week|day after tomorrow|in \d+ days)", query)
    if approximate_date_match:
        parsed_date = parse_approximate_date(approximate_date_match.group())
        if parsed_date:
            year, month, day = parsed_date.year, parsed_date.month, parsed_date.day
        else:
            day = today.day
    else:
        # ✅ Extract Specific Day (If Mentioned)
            # ✅ Extract Specific Day (If Mentioned)
        day_match = re.search(r"\b([1-9]|[12][0-9]|3[01])\b", query)
        day = int(day_match.group()) if day_match else None

        # ✅ If a Day Name (Friday, etc.) Exists, Align with the Correct Date
    if day_name:
        current_date = datetime.date(year, month, 1)  # Start from the 1st of the month
        while current_date.weekday() != days_map[day_name.lower()]:
            current_date += datetime.timedelta(days=1)  # Move to the next day
            day = current_date.day  # ✅ Assign the correct day based on the weekday name

        # ✅ If no specific day is found, default to today’s date
            # ✅ If no specific day is found, default to TODAY’s date
    if day is None:
        today = datetime.date.today()  # ✅ Get today’s date
        if month == today.month and year == today.year:
            day = today.day  # ✅ Keep today's actual date
        else:
            # ✅ If the user entered a different month/year, use today’s day but in that month/year
            try:
                day = min(today.day, (datetime.date(year, month, 1) + datetime.timedelta(days=31)).day)
            except ValueError:
                day = 1  # ✅ Handle invalid cases (e.g., February 30)

    # ✅ Get Correct Day Name
    day_of_week = datetime.date(year, month, day).weekday()
    day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day_of_week]



    # ✅ Convert to Day of the Week (Numeric)
    if day:
        day_of_week = datetime.date(year, month, day).weekday()
    else:
        day_of_week = today.weekday()
        day_name = today.strftime("%A")  # Default to today’s name if missing

    current_hour = datetime.datetime.now().hour 

    time_match = re.search(r'\b([0-9]{1,2}):?([0-9]{2})?\s?(a\.?m\.?|p\.?m\.?|am|pm)?\b', query)
    if time_match:
        hour = int(time_match.group(1))
        period = time_match.group(3)  # AM/PM format
        if period:
            period = period.replace(".", "").lower()  # Normalize "a.m." -> "am"
            if period == "pm" and hour < 12:
                hour += 12
            elif period == "am" and hour == 12:
                hour = 0  # Midnight case
    else:
        time_match = re.search(r'\b(morning|afternoon|evening|night)\b', query)
        hour_range = time_of_day_to_hour(time_match.group()) if time_match else None
        hour = hour_range[0] if hour_range else current_hour  # Default to system hour if missing

    # ✅ Determine Time of Day
    if 6 <= hour <= 10:
        time_of_day = "morning"
    elif 11 <= hour <= 15:
        time_of_day = "afternoon"
    elif 16 <= hour <= 19:
        time_of_day = "evening"
    elif 20 <= hour <= 23 or hour == 0:
        time_of_day = "night"
    else:
        time_of_day = "unspecified"
        
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
            
    return {
        "year": year,
        "month": month,
        "day_of_week": day_of_week,  # ✅ Still keeping the number
        "day_name": day_name,  # ✅ Now storing the actual day name
        "hour": hour,
        "time_of_day": time_of_day,
        
        "bird_name": bird_name if bird_name else "Unknown Bird"
    }

# API Endpoint for Birdwatching Prediction
@app.route('/predict_location', methods=['POST'])
def predict_best_locations():
    
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        logger.info(f"🔍 Received Query: {query}")
        
        features = extract_query_features(query)
        
        if features["bird_name"] == "Unknown Bird":
                return jsonify({
                    "message": "The query you entered didn't contain a bird species. Please select one and re-enter the query.",
                    "valid_bird_names": valid_bird_names
                })
        
        bird_name_encoded = label_encoders['COMMON NAME'].transform([features["bird_name"]])[0]
        
        results = []
        
        for location in predefined_locations:
            
            input_data = pd.DataFrame([[features["year"], features["month"], features["day_of_week"],
                                        features["hour"], location["LATITUDE"], location["LONGITUDE"],
                                        bird_name_encoded]],
                                    columns=selected_features)
            
            predicted_location_encoded = location_model.predict(input_data)[0]
            predicted_location = label_encoders['LOCALITY'].inverse_transform([predicted_location_encoded])[0]
            results.append(predicted_location)
        
        unique_locations = list(set(results))
        
    
        response = {
                "Response for you": f"The {features['bird_name']} can be seen "
               f"on {features['day_name']}, {features['month']}/{features['year']} "
               f"in the {features['time_of_day']} at these locations in Hambanthota District: {', '.join(unique_locations)}."
                }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"❌ Error in Prediction: {e}")
        return jsonify({"error": "Prediction error occurred"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
