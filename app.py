
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained ML model
model = joblib.load("model/iris_model.pkl")

# Flower names
species = ["setosa", "versicolor", "virginica"]


@app.route("/")
def home():
    return "MLOps Iris Prediction API is running!"


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    required_fields = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]

    if not data or not all(
        field in data for field in required_fields
    ):
        return jsonify({
            "error": "Please provide all four flower measurements."
        }), 400

    try:
        features = [[
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"])
        ]]
    except (TypeError, ValueError):
        return jsonify({
            "error": "Measurements must be numeric."
        }), 400

    prediction = int(model.predict(features)[0])

    return jsonify({
        "prediction": species[prediction]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
