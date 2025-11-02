import os
import time
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from Autocorrect_mod import *
from Autocomplete_mod import *

# --- Paths & setup ---
base_dir = os.path.abspath(os.path.dirname(__file__))
template_folder = os.path.join(base_dir, 'templates')
static_folder = os.path.join(base_dir, 'Static')
app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)

MODEL_DIR = os.path.join(base_dir, 'data')
MODEL_FILE1 = os.path.join(MODEL_DIR, "autocorrect_model_data.pkl")
MODEL_FILE2 = os.path.join(MODEL_DIR, "autocomplete_model_data.pkl")

# --- Model loading ---
vocab, probs = set(), {}
vocabulary, n_gram_counts_list = set(), []

try:
    print("Loading Autocorrect model...")
    vocab, probs = load_model_autocorrect(MODEL_FILE1)
    print(f"Autocorrect model loaded. Vocab size: {len(vocab)}")
except Exception as e:
    print(f"Error loading autocorrect model: {e}")

try:
    print("Loading Autocomplete model...")
    vocabulary, n_gram_counts_list = load_model(MODEL_FILE2)
    print(f"Autocomplete model loaded. Vocabulary size: {len(vocabulary)}")
except Exception as e:
    print(f"Error loading autocomplete model: {e}")

# --- Core functions ---
def autocorrect(word):
    if not vocab or not probs:
        return []
    return get_corrections_by_med(word.lower(), probs, vocab=vocab, n=3, verbose=False, display_matrix=False)[:3]


def generate_autocomplete(prefix):
    """Predict the next possible word(s) after the current sequence."""
    if not prefix.strip():
        return []
    tokens = prefix.lower().split()
    # Predict *next* words, not words starting with the last token
    suggestions_with_probs = get_suggestions(tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None)
    return [s[0] for s in suggestions_with_probs[:5]]

# --- Routes ---
@app.route("/")
def index():
    return render_template("Front.html")


@app.route("/autocorrect", methods=["GET"])
def autocorrect_api():
    word = request.args.get("word", "")
    if not word:
        return jsonify({"suggestions": []}), 200
    suggestions = autocorrect(word)
    return jsonify({"suggestions": suggestions}), 200


@app.route("/autocomplete", methods=["GET"])
def autocomplete_api():
    prefix = request.args.get("prefix", "")
    if not prefix:
        return jsonify({"suggestions": []}), 200
    predictions = generate_autocomplete(prefix)
    return jsonify({"suggestions": predictions}), 200


if __name__ == "__main__":
    app.run(debug=True, threaded = True)
