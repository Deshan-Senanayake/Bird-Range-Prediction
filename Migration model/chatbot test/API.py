from flask import Flask, request, jsonify
from predict_bird_presence import predict_bird_presence
from predict_best_locations import predict_best_locations
from predict_best_time import predict_best_time

app = Flask(__name__)

@app.route("/predict_presence", methods=["POST"])
def predict_presence():
    data = request.json
    query = data.get("query", "")
    result = predict_bird_presence(query)
    return jsonify({"response": result})

@app.route("/predict_location", methods=["POST"])
def predict_location():
    data = request.json
    query = data.get("query", "")
    result = predict_best_locations(query)
    return jsonify({"response": result})

@app.route("/predict_time", methods=["POST"])
def predict_time():
    data = request.json
    query = data.get("query", "")
    result = predict_best_time(query)
    return jsonify({"response": result})

if __name__ == "__main__":
    app.run(debug=True)
