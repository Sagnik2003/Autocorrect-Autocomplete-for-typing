import re
from collections import Counter
import numpy as np
import pandas as pd
import os
import pickle
import sys
import time

# --- CONFIGURATION ---
TRAINING_MODE = False  # Set to True once to train and save; Set to False for deployment
MODEL_FILE = "Autocorrect-Autocomplete-for-typing/App/data/autocorrect_model_data.pkl"
# ---------------------

# --- NEW: Model Persistence Functions ---

def save_model_autocorrect(vocab, probs, filename=MODEL_FILE):
    """Saves vocabulary set and probability dictionary to a pickle file."""
    data = {
        'vocab': vocab,
        'probs': probs
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nAutocorrect model data successfully saved to {filename}")

def load_model_autocorrect(filename=MODEL_FILE):
    """Loads vocabulary set and probability dictionary from a pickle file."""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"\nAutocorrect model data successfully loaded from {filename}")
        return data['vocab'], data['probs']
    except FileNotFoundError:
        print(f"Error: Model file {filename} not found. Must train the model first.")
        return None, None

# --- Core Autocorrect Functions ---

def process_data(file_name):
    """Reads file, converts to lowercase, and tokenizes into a list of words."""
    words = []
    try:
        with open(file_name, 'r',encoding='utf-8') as file:
            file_content = file.read()
    except FileNotFoundError:
        print(f"Error: Training file not found at {file_name}")
        return []
        
    file_content = file_content.lower()
    words = re.findall(r'\w+', file_content )
    return words

def get_count(word_l):
    """
    MODIFIED: Counts word frequencies using collections.Counter for O(N) efficiency.
    (Original was O(V * N) which is much slower for large corpora).
    """
    return Counter(word_l)
    
def get_probs(word_count_dict):
    """Calculates word probabilities."""
    probs = {}
    M = sum(word_count_dict.values())
    for word in word_count_dict.keys():
        probs[word] = word_count_dict[word] / M
    return probs

def delete_letter(word,verbose = False):
    """Returns a list of all words with one letter deleted."""
    delete_l = []
    split_l = [(word[:i], word[i:]) for i in range(len(word)+1)]
    delete_l = [a+b[1:] for a,b in split_l if b]
    
    if verbose:
        print(f"delete_letter('{word}')")
        print(f"split_l: {split_l}")
        print(f"delete_l: {delete_l}")
    return delete_l

def switch_leter(word,verbose = False):
    """Returns a list of all words with two adjacent letters swapped."""
    switch_l = []
    split_l = [(word[:i], word[i:]) for i in range(len(word)-1)]
    switch_l = [a + b[1] + b[0] + b[2:] for a,b in split_l]
    
    if verbose:
        print(f"switch_letter('{word}')")
        print(f"split_l: {split_l}")
        print(f"switch_l: {switch_l}")
    return switch_l

def replace_letter(word,verbose = False):
    """Returns a set of all words with one letter replaced by any other letter."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    replace_l = set()
    split_l = [(word[:i], word[i:]) for i in range(len(word))]
    
    for a,b in split_l:
        if b:
            for l in letters:
                new_word = a + l + b[1:]
                if new_word != word: # Exclude the original word
                     replace_l.add(new_word)
    
    replace_l = sorted(list(replace_l))
    
    if verbose:
        print(f"replace_letter('{word}')")
        print(f"replace_l: {replace_l}")
    return replace_l

def insert_letter(word,verbose = False):
    """Returns a list of all words with one letter inserted at any position."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    insert_l = []
    split_l = [(word[:i], word[i:]) for i in range(len(word)+1)]
    insert_l = [a + l + b for a,b in split_l for l in letters]
    
    if verbose:
        print(f"insert_letter('{word}')")
        print(f"insert_l: {insert_l}")
    return insert_l

def edit_one_letter(word,allow_switches = True):
    """Returns a set of all strings that are one edit away from 'word'."""
    edit_one_set = set()
    edit_one_set.update(delete_letter(word))
    edit_one_set.update(replace_letter(word))
    edit_one_set.update(insert_letter(word))
    if allow_switches:
        edit_one_set.update(switch_leter(word))
        
    return edit_one_set

def edit_two_letters(word, allow_switches = True):
    """Returns a set of all strings that are two edits away from 'word'."""
    edit_two_set = set()
    edit_one = edit_one_letter(word,allow_switches = allow_switches)
    for w in edit_one:
        if w:
            edit_two = edit_one_letter(w,allow_switches= allow_switches)
            edit_two_set.update(edit_two)
    return set(edit_two_set)

def min_edit_distance(source,target, ins_cost = 1, del_cost = 1, rep_cost = 2):
    """Calculates the Minimum Edit Distance (Levenshtein distance with custom costs)."""
    m = len(source)
    n = len(target)
    
    D = np.zeros((m+1,n+1), dtype=int)
    
    for row in range(1,m+1):
        D[row,0] = D[row-1,0] + del_cost
        
    for col in range(1,n+1):
        D[0,col] = D[0,col-1] + ins_cost
    
    for row in range(1,m+1):
        for col in range(1,n+1):
            r_cost = rep_cost
            if source[row-1] == target[col-1]:
                r_cost = 0
            D[row,col] = min(D[row-1,col] + del_cost,
                             D[row,col-1] + ins_cost,
                             D[row-1,col-1] + r_cost)
    med = D[m,n]
    return D, med

def get_corrections_by_med(word, probs, vocab, n=3, verbose = True, display_matrix = False):
    """
    Generates autocorrection suggestions by checking edit distance 1 and 2,
    then sorts by MED (ascending) and probability (descending).
    """
    suggestions_set = set()
    
    # 1. Check if word is already correct
    if word in vocab:
        suggestions_set.add(word)
    
    # 2. Check edit distance 1
    suggestions_set.update(edit_one_letter(word).intersection(vocab))
    
    # 3. Check edit distance 2 (only if no suggestions found in step 1 or 2)
    if not suggestions_set:
        suggestions_set.update(edit_two_letters(word).intersection(vocab))
        
    suggestions = list(suggestions_set)

    med_list = []
    for s in suggestions:
        # Calculate MED only for valid suggestions
        D, med = min_edit_distance(word, s)
        med_list.append((s, med, probs.get(s,0)))
    
    # Sort by min edit distance first (x[1] ascending), then by probability (-x[2] descending)
    med_list = sorted(med_list, key=lambda x: (x[1], -x[2]))
    
    n_best = med_list[:n]
    
    # Filter for output: only the word itself
    autocorrected_words = [w[0] for w in n_best]
    
    if verbose:
        print("entered word:", word)
        print("suggestions:", autocorrected_words)
    
    if display_matrix:
        # Helper function for displaying matrix (not fully provided but included here for completeness)
        pass 
        
    return autocorrected_words

if __name__ == "__main__":
    
    if TRAINING_MODE:
        print("--- Starting Autocorrect Model Training Mode ---")
        st = time.time()
        
        file_path = 'Autocorrect and Autocomplete/App/data/AllCombined.txt'
        word_l = process_data(file_path)
        
        if not word_l:
            sys.exit(1)
            
        vocab = set(word_l)
        word_count_dict = get_count(word_l) # FAST Counter
        probs = get_probs(word_count_dict)
        
        et = time.time()
        print(f"Training Time (Processing + Counting + Probs): {et-st:.4f}s")
        
        # --- SAVE THE MODEL ---
        save_model_autocorrect(vocab, probs, MODEL_FILE)
        # ----------------------
        
    # --- PREDICTION / DEPLOYMENT MODE ---
    else:
        print("--- Starting Prediction Mode (Loading Model) ---")
        st = time.time()
        vocab, probs = load_model_autocorrect(MODEL_FILE)
        
        if not vocab:
            print("Cannot run in prediction mode without a trained model file.")
            sys.exit(1)
            
        et = time.time()
        print(f"Model loading time: {et-st:.4f}s (Extremely fast for web app startup!)")
        
        # Example usage for Autocorrect
        word = input("\nEnter a word for autocorrection: ")
        
        sst = time.time()
        corrections = get_corrections_by_med(word, probs, vocab, n=3, verbose=True, display_matrix=False)
        tt = time.time()
        
        print(f"Correction Time: {tt-sst:.4f}s")
        print(f"Top 3 Corrections for '{word}': {corrections}")
        
        
        
