import joblib
import pandas as pd
import numpy as np
import re
import datetime
import logging
from flask import Flask, request, jsonify
from difflib import get_close_matches
from rapidfuzz import process
from dateutil import parser
from flask_cors import CORS

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)

# ✅ Set up Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load Model from GitHub
import requests
import joblib
import io
import os

MODEL_URL1 = "https://raw.githubusercontent.com/Deshan-Senanayake/Bird-Range-Prediction/main/Migration%20model/models/migration_prediction_model.pkl"

# ✅ Define a local model path
MODEL_PATH1 = "migration_prediction_model.pkl"

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
model_file_path = download_model(MODEL_URL1, MODEL_PATH1)

# ✅ Load Model
with open(model_file_path, "rb") as model_file:
    model_data1 = joblib.load(model_file)

rf_model = model_data1['rf_final']
label_encoders1 = model_data1['label_encoders']
selected_features1 = model_data1['selected_features']

print("✅ Model loaded successfully!")

# ✅ Valid Localities & Bird Names
valid_localities = [
    "Buckingham Place Hotel Tangalle", "Bundala NP General", "Bundala National Park",
    "Kalametiya", "Tissa Lake", "Yala National Park General", "Debarawewa Lake", 
    
   
    
]

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

# ✅ Function: Correct Bird Name
def correct_bird_name(name):
    name = name.lower()
    if name in bird_aliases:
        return bird_aliases[name]
    matches = get_close_matches(name, [b.lower() for b in valid_bird_names], n=1, cutoff=0.3)
    return next((b for b in valid_bird_names if b.lower() == matches[0]), "Unknown Bird")

# ✅ Function: Correct Locality
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

# ✅ Function: Convert Day Name to Integer
def day_name_to_int(day_name):
    days_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }
    return days_map.get(day_name.lower(), None)

# ✅ Function: Convert Time of Day to Hour
def time_of_day_to_hour(time_str):
    time_ranges = {"morning": (6, 10), "afternoon": (11, 15), "evening": (16, 19), "night": (20, 23)}
    return time_ranges.get(time_str.lower(), None)

# ✅ Function: Parse Approximate Date
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

# ✅ Function: Extract Features from Query
def extract_query_features_bird_presence(query):
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




    # ✅ Extract Locality Properly
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

    # ✅ Extract Bird Name Properly
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
        "locality": locality if locality else "Unknown Location",
        "bird_name": bird_name if bird_name else "Unknown Bird"
    }



# ✅ API Route: Prediction
@app.route("/predict_presence", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        logger.info(f"🔍 Received Query: {query}")

        if not query:
            return jsonify({"error": "No query provided"}), 400

        features = extract_query_features_bird_presence(query)

        # ✅ Check if Locality is Missing
        if features["locality"] == "Unknown Location":
            return jsonify({
                "message": "The query you entered didn't contain a location in Hambanthota District. Please select one and re-enter the query.",
                "you can use these locations": [
                    "You can use 'Bundala' instead of 'Bundala NP General'.",
                    "You can use 'Yala' instead of 'Yala National Park General'.",
                    "You can use 'Tissa' instead of 'Tissa Lake'."
                ]
            })

        # ✅ Check if Bird Name is Missing
        if features["bird_name"] == "Unknown Bird":
            return jsonify({
                "message": "The query you entered didn't contain a bird species. Please select one and re-enter the query.",
                "valid_bird_names": valid_bird_names
            })

        # ✅ Encode Locality & Bird Name
        locality_encoded = label_encoders1['LOCALITY'].transform([features["locality"]])[0]
        bird_name_encoded = label_encoders1['COMMON NAME'].transform([features["bird_name"]])[0]

        # ✅ Prepare Input Data
        input_data = pd.DataFrame([[features["year"], features["month"], features["day_of_week"],
                                    features["hour"], locality_encoded, bird_name_encoded]], 
                                   columns=selected_features1)

        # ✅ Make Prediction
        probability = rf_model.predict_proba(input_data)[:, 1][0]
        prediction = int(probability >= 0.5)

        # ✅ Construct Response
        # ✅ Construct Response with Day Name
        response = {
         
            "Response": (
                f"The {features['bird_name']} is {'likely' if 1 else 'unlikely'} "
                f"to be present at {features['locality']} on {features['day_name']}, {features['month']}/{features['year']} "
                f"in the {features['time_of_day']}."
            )
        }


        return jsonify(response), 200

    except Exception as e:
        logger.error(f"❌ Error in Prediction: {e}")
        return jsonify({"error": "Prediction error occurred"}), 500
    
    
    
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
MODEL_URL2 = "https://raw.githubusercontent.com/Deshan-Senanayake/Bird-Range-Prediction/main/Migration%20model/models/location_prediction_model.pkl"

# ✅ Define a local model path
MODEL_PATH2 = "location_prediction_model.pkl"

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
model_file_path2 = download_model(MODEL_URL2, MODEL_PATH2)

# ✅ Load Model
with open(model_file_path2, "rb") as model_file2:
    model_data2 = joblib.load(model_file2)

location_model2 = model_data2['location_model']
selected_features2 = model_data2['selected_features']
label_encoders2 = model_data2['label_encoders']

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
        
        features2 = extract_query_features(query)
        
        if features2["bird_name"] == "Unknown Bird":
                return jsonify({
                    "message": "The query you entered didn't contain a bird species. Please select one and re-enter the query.",
                    "valid_bird_names": valid_bird_names
                })
        
        bird_name_encoded = label_encoders2['COMMON NAME'].transform([features2["bird_name"]])[0]
        
        results = []
        
        for location in predefined_locations:
            
            

            
            input_data = pd.DataFrame([[features2["year"], features2["month"], features2["day_of_week"],
                                        features2["hour"], location["LATITUDE"], location["LONGITUDE"],
                                        bird_name_encoded]],
                                    columns=selected_features2)
            
            predicted_location_encoded = location_model2.predict(input_data)[0]
            predicted_location = label_encoders2['LOCALITY'].inverse_transform([predicted_location_encoded])[0]
            results.append(predicted_location)
        
        unique_locations = list(set(results))
        
    
        response = {
                "Response for you": f"The {features2['bird_name']} can be seen "
               f"on {features2['day_name']}, {features2['month']}/{features2['year']} "
               f"in the {features2['time_of_day']} at these locations in Hambanthota District: {', '.join(unique_locations)}."
                }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.exception("❌ Exception in /predict_location:")
        return jsonify({"error": f"Prediction error occurred: {str(e)}"}), 500

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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load Model & Encoders from GitHub
import requests
import joblib
import io
import os

MODEL_URL3 = "https://raw.githubusercontent.com/Deshan-Senanayake/Bird-Range-Prediction/main/Migration%20model/models/time_prediction_model.pkl"

# ✅ Define a local model path
MODEL_PATH3 = "time_prediction_model.pkl"

# ✅ Function: Download Large Model in Chunks
def download_model(url3, save_path3, chunk_size=1024 * 1024):  # 1MB chunks
    if os.path.exists(save_path3):  # ✅ Skip download if file exists
        print(f"📁 Using cached model: {save_path3}")
        return save_path3

    print("📥 Downloading model. Please wait...")
    
    with requests.get(url3, stream=True) as response3:
        response3.raise_for_status()  # ✅ Check for errors
        total_size = int(response3.headers.get("content-length", 0))  # ✅ Get file size
        downloaded = 0
        
        with open(save_path3, "wb") as file:
            for chunk in response3.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    print(f"🔄 Downloaded: {downloaded / total_size:.2%}", end="\r")  # ✅ Show progress

    print("\n✅ Model downloaded successfully.")
    return save_path3

# ✅ Download model if not cached
model_file_path3 = download_model(MODEL_URL3, MODEL_PATH3)

# ✅ Load Model
with open(model_file_path3, "rb") as model_file3:
    model_data3 = joblib.load(model_file3)

month_model = model_data3['month_model']
hour_model = model_data3['hour_model']
selected_features3 = model_data3['selected_features']
label_encoders3 = model_data3['label_encoders']

print("✅ Model loaded successfully!")


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

            locality_encoded = label_encoders3['LOCALITY'].transform([features["locality"]])[0]
            bird_name_encoded = label_encoders3['COMMON NAME'].transform([features["bird_name"]])[0]

        except ValueError as e:
            logger.error(f"Encoding Error: {e}")
            return jsonify({"error": f"Invalid input detected: {str(e)}"}), 400  # ✅ Return proper error message



        input_data = pd.DataFrame([[1, features["year"], features["day_of_week"],
                                    locality_encoded, bird_name_encoded,
                                    features["Is_Summer"], features["Is_Winter"], features["Is_Spring"], features["Is_Autumn"],
                                    features["Is_Morning"], features["Is_Afternoon"], features["Is_Evening"], features["Is_Night"]]],
                                columns=selected_features3)

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



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

