import gradio as gr
import requests

# Flask API Base URL
API_URL = "http://127.0.0.1:5000"

# Function to send user input to the appropriate model
def chatbot_response(user_input):
    # Determine which model to use based on the input
    if "where" in user_input.lower():
        response = requests.post(f"{API_URL}/predict_location", json={"query": user_input})
    elif "when" in user_input.lower():
        response = requests.post(f"{API_URL}/predict_time", json={"query": user_input})
    else:
        response = requests.post(f"{API_URL}/predict_presence", json={"query": user_input})

    result = response.json()
    return result.get("response", "I'm sorry, I couldn't understand your query.")

# Create the Gradio Interface
chatbot = gr.ChatInterface(chatbot_response, title="Birdwatching Chatbot")

# Launch the Chatbot
chatbot.launch()
