#earlier one


import requests
import logging
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

logger = logging.getLogger(__name__)

# API endpoint
RANGE_PREDICTION_API = "http://127.0.0.1:5000/predict_presence"
LOCATION_API = "http://127.0.0.1:5001/predict_location"
TIME_PREDICTION_API = "http://127.0.0.1:5002/predict_best_time"

class ActionHandleBirdPrediction(Action):
    def name(self) -> str:
        return "action_range_prediction"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        # ✅ Get user query
        user_query = tracker.latest_message.get("text", "").strip()
        logger.info(f"🔍 User Query Sent to API: {user_query}")

        if not user_query:
            dispatcher.utter_message(text="I couldn't understand your request. Could you rephrase it?")
            return []

        # ✅ API Request
        request_payload = {"query": user_query}
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(RANGE_PREDICTION_API, json=request_payload, headers=headers)
            json_response = response.json()

            # ✅ Handle Missing Locality
            if "valid_localities" in json_response:
                valid_locations_text = "\n".join(json_response["valid_localities"])
                location_aliases_text = "\n".join(json_response.get("location_aliases", []))
                dispatcher.utter_message(
                    text=f"{json_response['message']}\n\nValid Locations:\n{valid_locations_text}\n\n{location_aliases_text}"
                )
                return []

            # ✅ Handle Missing Bird Name
            if "valid_bird_names" in json_response:
                valid_birds_text = "\n".join(json_response["valid_bird_names"])
                dispatcher.utter_message(
                    text=f"{json_response['message']}\n\nValid Bird Species:\n{valid_birds_text}"
                )
                return []

            # ✅ Extract Response Field
            meaningful_sentence = json_response.get("Response", "I couldn't generate a response.")
            dispatcher.utter_message(text=meaningful_sentence)

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API call error: {e}")
            dispatcher.utter_message(text="There was an error connecting to the prediction API.")

        return []

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        # ✅ Get user query
        user_query = tracker.latest_message.get("text", "").strip()
        logger.info(f"🔍 User Query Sent to Birdwatching API: {user_query}")

        if not user_query:
            dispatcher.utter_message(text="I couldn't understand your request. Could you rephrase it?")
            return []

        # ✅ API Request
        request_payload = {"query": user_query}
        headers = {"Content-Type": "application/json"}
        LOCATION_API = "http://127.0.0.1:5001/predict_location"  # ✅ Update API Endpoint

        try:
            response = requests.post(LOCATION_API, json=request_payload, headers=headers)
            json_response = response.json()

            # ✅ Handle Missing Bird Name
            if "valid_bird_names" in json_response:
                valid_birds_text = "\n".join(json_response["valid_bird_names"])
                dispatcher.utter_message(text=f"{json_response['message']}\n\nValid Bird Species:\n{valid_birds_text}")
                return []

            # ✅ Handle Prediction Result
            message = json_response.get("message", "I couldn't generate a response.")
            dispatcher.utter_message(text=message)

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API call error: {e}")
            dispatcher.utter_message(text="There was an error connecting to the birdwatching location prediction API.")

        return []


    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        TIME_PREDICTION_API = "http://127.0.0.1:5002/predict_best_time"  # ✅ API Endpoint

        # ✅ Get user query
        user_query = tracker.latest_message.get("text", "").strip()
        logger.info(f"🔍 User Query Sent to Time Prediction API: {user_query}")

        if not user_query:
            dispatcher.utter_message(text="I couldn't understand your request. Could you rephrase it?")
            return []

        # ✅ API Request
        request_payload = {"query": user_query}
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(TIME_PREDICTION_API, json=request_payload, headers=headers)

            # ✅ Handle HTTP errors
            if response.status_code != 200:
                logger.error(f"❌ API Error: {response.status_code} - {response.text}")
                dispatcher.utter_message(text="There was an error processing your request. Please try again.")
                return []

            json_response = response.json()

            # ✅ Handle Missing Locality
            if "valid_localities" in json_response:
                valid_locations_text = "\n".join(json_response["valid_localities"])
                dispatcher.utter_message(text=f"{json_response['message']}\n\nValid Locations:\n{valid_locations_text}")
                return []

            # ✅ Handle Missing Bird Name
            if "valid_bird_names" in json_response:
                valid_birds_text = "\n".join(json_response["valid_bird_names"])
                dispatcher.utter_message(text=f"{json_response['message']}\n\nValid Bird Species:\n{valid_birds_text}")
                return []

            # ✅ Handle Prediction Result
            month = json_response.get("month", None)
            hour = json_response.get("hour", None)
            bird_name = json_response.get("bird_name", None)
            locality = json_response.get("locality", None)
            day_name = json_response.get("day_name", None)

            if None in [month, hour, bird_name, locality, day_name]:
                logger.error("❌ Missing required data in API response.")
                dispatcher.utter_message(text="There was an issue retrieving the prediction. Please try again.")
                return []

            dispatcher.utter_message(
                text=f"The best time to see the {bird_name} at {locality} on {day_name} is in {month} at {hour}."
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API call error: {e}")
            dispatcher.utter_message(text="There was an error connecting to the time prediction API.")

        return []



    
