import requests
import logging
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

logger = logging.getLogger(__name__)

# ✅ API Endpoints
RANGE_PREDICTION_API = "http://127.0.0.1:5000/predict_presence"
LOCATION_API = "http://127.0.0.1:5001/predict_location"
TIME_PREDICTION_API = "http://127.0.0.1:5002/predict_best_time"

# ✅ Function: Call Range Prediction API
def handle_range_prediction(query, dispatcher):
    payload = {"query": query}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(RANGE_PREDICTION_API, json=payload, headers=headers)
        json_response = response.json()

        if "valid_localities" in json_response:
            valid_locations_text = "\n".join(json_response["valid_localities"])
            location_aliases_text = "\n".join(json_response.get("location_aliases", []))
            dispatcher.utter_message(
                text=f"{json_response['message']}\n\nValid Locations:\n{valid_locations_text}\n\n{location_aliases_text}"
            )
            return

        if "valid_bird_names" in json_response:
            valid_birds_text = "\n".join(json_response["valid_bird_names"])
            dispatcher.utter_message(
                text=f"{json_response['message']}\n\nValid Bird Species:\n{valid_birds_text}"
            )
            return

        dispatcher.utter_message(text=json_response.get("Response", "I couldn't generate a response."))

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ API call error: {e}")
        dispatcher.utter_message(text="There was an error connecting to the range prediction API.")

# ✅ Function: Call Location Prediction API
def handle_location_prediction(query, dispatcher):
    payload = {"query": query}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(LOCATION_API, json=payload, headers=headers)
        json_response = response.json()

        if "valid_bird_names" in json_response:
            valid_birds_text = "\n".join(json_response["valid_bird_names"])
            dispatcher.utter_message(text=f"{json_response['message']}\n\nValid Bird Species:\n{valid_birds_text}")
            return

        dispatcher.utter_message(text=json_response.get("Response for you", "I couldn't generate a response."))

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ API call error: {e}")
        dispatcher.utter_message(text="There was an error connecting to the location prediction API.")

# ✅ Function: Call Time Prediction API
def handle_time_prediction(query, dispatcher):
    payload = {"query": query}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(TIME_PREDICTION_API, json=payload, headers=headers)

        if response.status_code != 200:
            logger.error(f"❌ API Error: {response.status_code} - {response.text}")
            dispatcher.utter_message(text="There was an error processing your request.")
            return

        json_response = response.json()

        if "valid_localities" in json_response:
            valid_locations_text = "\n".join(json_response["valid_localities"])
            dispatcher.utter_message(text=f"{json_response['message']}\n\nValid Locations:\n{valid_locations_text}")
            return

        if "valid_bird_names" in json_response:
            valid_birds_text = "\n".join(json_response["valid_bird_names"])
            dispatcher.utter_message(text=f"{json_response['message']}\n\nValid Bird Species:\n{valid_birds_text}")
            return

        dispatcher.utter_message(text=json_response.get("Response", "I couldn't generate a response."))
        

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ API call error: {e}")
        dispatcher.utter_message(text="There was an error connecting to the time prediction API.")

# ✅ Keyword Routing Logic
def determine_api_from_query(query: str) -> str:
    query = query.lower()

    time_keywords = ["best time", "morning", "afternoon", "evening", "night", "what time", "hour"]
    location_keywords = ["where", "location", "spot", "place", "area", "district"]
    presence_keywords = ["present", "see", "appear", "found", "visible"]

    if any(word in query for word in time_keywords):
        return "time"
    elif any(word in query for word in location_keywords):
        return "location"
    else:
        return "presence"  # Default fallback

# ✅ Rasa Action Class
class ActionHandleBirdPrediction(Action):
    def name(self) -> str:
        return "action_range_prediction"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_query = tracker.latest_message.get("text", "").strip()
        logger.info(f"🔍 Received user query: {user_query}")

        if not user_query:
            dispatcher.utter_message(text="I couldn't understand your request. Can you rephrase it?")
            return []

        selected_api = determine_api_from_query(user_query)
        logger.info(f"📡 Routing query to: {selected_api.upper()} API")

        if selected_api == "time":
            handle_time_prediction(user_query, dispatcher)
        elif selected_api == "location":
            handle_location_prediction(user_query, dispatcher)
        else:
            handle_range_prediction(user_query, dispatcher)

        return []
