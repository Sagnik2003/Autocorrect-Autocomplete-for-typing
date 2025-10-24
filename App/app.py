import os
from flask import Flask, request, jsonify, render_template
from autocomplete import *
import numpy as np
import time

# --- Fixes Start Here ---

# Get the absolute path of the directory this file (app.py) is in
base_dir = os.path.abspath(os.path.dirname(__file__))

# 1. Tell Flask where the 'templates' folder is
template_folder = os.path.join(base_dir, 'templates')

# 2. Tell Flask where the 'Static' folder is (since yours is capitalized)
static_folder = os.path.join(base_dir, 'Static')

# 3. Create the app instance with the correct paths
app = Flask(__name__, 
            template_folder=template_folder, 
            static_folder=static_folder)

# --- Fixes End Here ---


# Dummy corpus for suggestions
vocab = None
WORDS = None
word_count_dict = None
probs = None

def load_corpus():
    global vocab, WORDS, word_count_dict, probs
    if vocab is None:
        # 4. Fix the data file path to be relative to app.py
        print("Loading corpus...")
        A = time.time()
        data_path = os.path.join(base_dir, 'data', 'shakespeare.txt')
        B = time.time()
        print(f"Time to get data path: {B - A:.4f} seconds")
        
        vocab = process_data(data_path)
        WORDS = set(vocab)
        word_count_dict = get_count(vocab)
        probs = get_probs(word_count_dict)

def autocorrect(word):
    """
    Suggests possible corrections for a given word using Levenshtein distance and word probabilities.
    """
    load_corpus()
    suggestions = get_corrections_by_med(word, probs, WORDS, n=3, verbose=False, display_matrix=False)
    return suggestions

# # 5. Uncomment your autocomplete function (your JS needs it!)
# def autocomplete(prefix):
#     """
#     Returns a list of suggested word completions based on the given prefix.
#     """
#     # 🔹 Replace this with n-gram probabilities
#     if prefix.lower().endswith("i am"):
#         return ["happy", "tired", "learning"]
#     elif prefix.lower().endswith("hello"):
#         return ["world", "there", "everyone"]
#     else:
#         return ["test", "project", "fun"]

@app.route("/")
def index():
    return render_template("Front.html")

@app.route("/autocorrect")
def autocorrect_api():
    """
    API endpoint that returns autocorrect suggestions for a given word.
    Expects a 'word' query parameter and returns a JSON response with suggestions.
    """
    word = request.args.get("word", "")
    if not word:
        return jsonify({"error": "Missing 'word' parameter"}), 400
    suggestions = autocorrect(word)
    return jsonify(suggestions)

# # 6. Uncomment your autocomplete API route (your JS needs this!)
# @app.route("/autocomplete")
# def autocomplete_api():
#     prefix = request.args.get("prefix", "")
#     if not prefix:
#         return jsonify({"error": "Missing required parameter: prefix"}), 400
#     return jsonify(autocomplete(prefix))

if __name__ == "__main__":
    app.run(debug=True)