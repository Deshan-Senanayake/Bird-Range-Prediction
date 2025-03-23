from flask import Flask, request, jsonify
import joblib
import pandas as pd
import io
import requests
from rapidfuzz import process
import re
from flask_cors import CORS

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS) for Rasa

# ✅ Load the Trained Model from GitHub
url1 = "https://raw.githubusercontent.com/Deshan-Senanayake/Bird-Range-Prediction/main/Migration%20model/models/migration_prediction_model.pkl"
response1 = requests.get(url1)
response1.raise_for_status()  # Ensure request is successful
model_data1 = joblib.load(io.BytesIO(response1.content))

# ✅ Extract Model Components
rf_model = model_data1['rf_final']
label_encoders1 = model_data1['label_encoders']
selected_features1 = model_data1['selected_features']

# ✅ Define Valid Localities & Bird Names
valid_localities = [
    "Buckingham Place Hotel Tangalle", "Bundala NP General", "Bundala National Park",
    "Kalametiya", "Tissa Lake", "Yala National Park General", "Debarawewa Lake"
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

# ✅ Improved Locality Matching
def correct_locality(user_input):
    """ Matches the user's location input with the closest valid location. """
    best_match = process.extractOne(user_input, valid_localities, score_cutoff=75)
    return best_match[0] if best_match else "Unknown Location"

# ✅ Convert Day Name to Integer
def day_name_to_int(day_name):
    days_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, 
                "friday": 4, "saturday": 5, "sunday": 6}
    return days_map.get(day_name.lower(), None)

# ✅ Extract Information from Query
def extract_info_from_query(query):
    """ Extracts bird name, location, date, and time from the user's query. """
    if not query:
        return {"error": "No query provided."}

    extracted_data = {
        "bird_name": None, "locality": None, "year": 2025, "month": None,
        "day_of_week": None, "hour": None
    }
    
    import calendar

    months_map = {month.lower(): index for index, month in enumerate(calendar.month_name) if month}

    for word in query.split():
        if word.lower() in months_map:
            extracted_data["month"] = months_map[word.lower()]


    # Extract location
    for loc in valid_localities:
        if loc.lower() in query.lower():
            extracted_data["locality"] = loc
            break

    if not extracted_data["locality"]:
        words = query.split()
        for word in words:
            matched_location = correct_locality(word)
            if matched_location != "Unknown Location":
                extracted_data["locality"] = matched_location
                break

    # Extract bird name
    for word in query.split():
        if word.lower() in bird_aliases:
            extracted_data["bird_name"] = bird_aliases[word.lower()]
            break  # Stop after first match

    # If not found in aliases, check against valid bird names
    if not extracted_data["bird_name"]:
        for bird in valid_bird_names:
            if bird.lower() in query.lower():
                extracted_data["bird_name"] = bird
                break

    # Extract year
    match = re.search(r"\b(20\d{2})\b", query)
    if match:
        extracted_data["year"] = int(match.group(1))

    # Extract time (handles "10 AM", "2:30 p.m.", "14:00", "6:15am")
    match = re.search(r"\b(\d{1,2}(:\d{2})? ?[ap]m|\d{1,2}(:\d{2})?)\b", query.lower())
    if match:
        time_value = match.group(1).replace(" ", "").lower()
        if "am" in time_value or "pm" in time_value:
            extracted_data["hour"] = int(re.search(r"\d+", time_value).group())
            if "pm" in time_value and extracted_data["hour"] != 12:
                extracted_data["hour"] += 12
        else:
            extracted_data["hour"] = int(time_value.split(":")[0])

    # Extract day of the week
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
        if day in query.lower():
            extracted_data["day_of_week"] = day_name_to_int(day)
            break

    return extracted_data

# ✅ Prediction Function
def predict_bird_presence(features):
    """ Predicts bird presence based on extracted features. """
    if features["locality"] == "Unknown Location":
        return {"error": "Location not recognized."}

    if features["bird_name"] not in label_encoders1['COMMON NAME'].classes_:
        return {"error": "Bird name not recognized."}

    try:
        locality_encoded = label_encoders1['LOCALITY'].transform([features["locality"]])[0]
        bird_name_encoded = label_encoders1['COMMON NAME'].transform([features["bird_name"]])[0]

        input_data = pd.DataFrame([[features["year"], features["month"], features["day_of_week"],
                                     features["hour"], locality_encoded, bird_name_encoded]],
                                  columns=selected_features1)

        probability = rf_model.predict_proba(input_data)[:, 1][0]
        prediction = int(probability >= 0.5)

        return {
            "predicted_presence": prediction,
            "probability": round(probability, 3),
            "meaningful_sentence": (
                f"The {features['bird_name']} is likely to be present at {features['locality']} on {features['day_of_week']}, "
                f"{features['month']}/{features['year']} at {features['hour']}:00. (Confidence: {probability:.1f}%)"
            )
        }

    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

# ✅ Define Flask API Route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "No query provided."}), 400

    extracted_data = extract_info_from_query(query)

    if extracted_data.get("error"):
        return jsonify(extracted_data), 400

    prediction_result = predict_bird_presence(extracted_data)

    return jsonify(prediction_result)

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
