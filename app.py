"""
Signal — Spam Detector API
---------------------------
A small Flask REST API that wraps the spaCy text-classification model
trained in spammer.ipynb and serves it to the Signal frontend
(spam-classifier.html).

Endpoints:
    GET  /health            -> quick check that the server + model are up
    POST /predict            -> classify a message as spam or authentic

Run:
    python app.py
    (defaults to http://localhost:5000)
"""

import os


import spacy
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "fake_review_model")
)

print("Model path:", MODEL_PATH)
print("Exists:", os.path.exists(MODEL_PATH))

nlp = spacy.load(MODEL_PATH)
app = Flask(__name__)
CORS(app)  # allow the HTML page (opened from anywhere) to call this API

print(f"Loading model from {MODEL_PATH} ...")
nlp = spacy.load(MODEL_PATH)
print("Model loaded.")


def score_text(text):
    """
    Runs the model on a message and turns spaCy's raw output into a
    simple, frontend-friendly shape.

    The notebook trained the model with this label mapping:
        REAL -> spam probability
        FAKE -> ham / authentic probability
    (an unusual naming choice from the original notebook, kept as-is so the
    model doesn't need to be retrained.)
    """
    doc = nlp(text)
    spam_prob = float(doc.cats.get("REAL", 0.0))
    ham_prob = float(doc.cats.get("FAKE", 0.0))

    spam_percent = round(spam_prob * 100)

    if spam_percent >= 62:
        label = "spam"
        description = (
            "Multiple manipulation patterns detected — urgency, bait, or "
            "trigger phrases typical of unsolicited mail."
        )
    elif spam_percent >= 35:
        label = "uncertain"
        description = (
            "Some spam-like signals present, but not conclusive. Worth a "
            "closer read before trusting it."
        )
    else:
        label = "authentic"
        description = (
            "Reads like a genuine message — plain tone, few pressure "
            "tactics, no obvious bait."
        )

    return {
        "label": label,
        "spam_score": spam_percent,
        "description": description,
        "raw": {"spam_probability": spam_prob, "authentic_probability": ham_prob},
    }


@app.get("/")
def home():
    # Serves the Signal frontend directly, so you can just visit
    # http://localhost:5000 and skip opening the HTML file separately.
    return send_from_directory(".", "index.html")
    


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Field 'text' is required and cannot be empty."}), 400

    result = score_text(text)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
