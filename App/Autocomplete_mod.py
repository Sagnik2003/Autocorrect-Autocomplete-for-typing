import math
import random
import numpy as np
import pandas as pd
import nltk
import time
import sys
import pickle
import os

# Ensure nltk data is available (if necessary, uncomment the download line)
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt')
from nltk.tokenize import word_tokenize


# --- CONFIGURATION ---
# Set to True once to train and save; False for deployment.
TRAINING_MODE = False
# Using a relative path for the model file for better cross-platform compatibility.
MODEL_FILE = r"Autocorrect-Autocomplete-for-typing/App/data/autocomplete_model_data.pkl" 
# Use a raw string for the file path to avoid 'invalid escape sequence' warnings/errors on Windows.
# NOTE: You should update this path to where your 'AllCombined.txt' file is located.
TRAIN_DATA_PATH = r"Autocorrect-Autocomplete-for-typing/App/data/AllCombined.txt"

# --- 1. Data Preprocessing Functions ---

def save_model(vocabulary, n_gram_counts_list, filename=MODEL_FILE):
    """Saves vocabulary set and N-gram counts to a pickle file."""
    data = {
        'vocabulary': vocabulary,
        'n_gram_counts_list': n_gram_counts_list
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nAutocomplete model data successfully saved to {filename}")

def load_model(filename=MODEL_FILE):
    """Loads vocabulary set and N-gram counts from a pickle file."""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"\nAutocomplete model data successfully loaded from {filename}")
        return data['vocabulary'], data['n_gram_counts_list']
    except FileNotFoundError:
        print(f"Error: Model file {filename} not found. Must train the model first.")
        return None, None
    except KeyError as e:
        # Fixed: The KeyError was due to an old model format. Ensure the model is re-trained.
        print(f"Error loading model: Key {e} not found in the model file. Please ensure the model was trained with the current format (TRAINING_MODE=True).")
        return None, None

def split_to_sentences(data):
    """Splits text data into a list of sentences."""
    sentences = data.split("\n") 
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]
    return sentences

def tokenize_sentences(sentences):
    """Tokenizes a list of sentence strings into a list of token lists."""
    tokenized_list = []
    for sentence in sentences:
        sentence = sentence.lower()
        # Use word_tokenize from nltk
        tokenized = word_tokenize(sentence)
        # --- FIX: Filter out tokens that are purely punctuation or digits ---
        cleaned_tokens = [token for token in tokenized if token.isalpha() or token.isalnum() or "'" in token or "-" in token]
        # Fallback to keep tokens if alpha/alnum filter is too strict, but try to exclude common punctuation
        if not cleaned_tokens:
             cleaned_tokens = [token for token in tokenized if token not in ('.', ',', '!', '?', ':', ';', '(', ')', '[', ']', '{', '}', '"', "'")]
             
        tokenized_list.append(cleaned_tokens)
        # --- END FIX ---
    return tokenized_list

def tokenize_data(data):
    """Tokenizes raw text data into a list of tokenized sentences."""
    sentences = split_to_sentences(data)
    tokenized_sentences = tokenize_sentences(sentences)
    return tokenized_sentences

def count_words(tokenized_sentences):
    """Counts the frequency of each word in the tokenized data."""
    word_counts = {}
    for sentence in tokenized_sentences:
        for token in sentence:
            word_counts[token] = word_counts.get(token, 0) + 1
    return word_counts

def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    """Creates a vocabulary of words appearing at least count_threshold times."""
    word_counts = count_words(tokenized_sentences)
    closed_vocab = [word for word, count in word_counts.items() if count >= count_threshold]
    return closed_vocab

def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<UNK>"):
    """Replaces words not in the vocabulary (OOV) with the UNK token."""
    vocabulary = set(vocabulary)
    updated_tokenized_sentences = []
    for sentence in tokenized_sentences:
        updated_sentence = []
        for token in sentence:
            if token in vocabulary:
                updated_sentence.append(token)
            else:
                updated_sentence.append(unknown_token)
        updated_tokenized_sentences.append(updated_sentence)
    return updated_tokenized_sentences

def preprocess_data(train_data, test_data, count_threshold, unknown_token = "<UNK>", get_words_with_nplus_frequency_fn = get_words_with_nplus_frequency):
    """Orchestrates the entire data preprocessing pipeline."""
    tokenized_train_sentences = tokenize_data(train_data)
    vocabulary = get_words_with_nplus_frequency_fn(tokenized_train_sentences, count_threshold)
    
    tokenized_train_sentences = replace_oov_words_by_unk(tokenized_train_sentences, vocabulary, unknown_token)
    
    tokenized_test_sentences = tokenize_data(test_data)
    tokenized_test_sentences = replace_oov_words_by_unk(tokenized_test_sentences, vocabulary, unknown_token)
    
    return tokenized_train_sentences, tokenized_test_sentences, vocabulary

# --- 2. N-gram Counting and Probability Estimation ---

def count_n_grams(data, n , start_token = '<s>', end_token = '</s>'):
    """Counts N-grams in the tokenized data, adding start/end tokens."""
    n_gram_counts = {}
    for sentence in data:
        # Pad sentence with start and end tokens
        sentence = [start_token] * (n - 1) + sentence + [end_token]
        L  = len(sentence)
        m = L - n + 1
        for i in range(m):
            n_gram = tuple(sentence[i:i+n])
            n_gram_counts[n_gram] = n_gram_counts.get(n_gram, 0) + 1
    return n_gram_counts

def estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k = 1.0):
    """Estimates the smoothed probability P(word | previous_n_gram) using Laplace smoothing."""
    previous_n_gram = tuple(previous_n_gram)
    
    # Context count (N-gram count)
    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)
    denominator = previous_n_gram_count + k * vocabulary_size
    
    # Context + Word count ((N+1)-gram count)
    n_plus1_gram = previous_n_gram + (word,)
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)
    
    numerator = n_plus1_gram_count + k
    probability = numerator / denominator
    return probability

def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='</s>', unknown_token="<UNK>", k=1.0):
    """Calculates smoothed probabilities for all words in the vocabulary."""
    previous_n_gram = tuple(previous_n_gram)     
    
    # Add special tokens to vocabulary for size calculation
    # NOTE: Assumes vocabulary passed is a list of unique words
    vocabulary_plus_special = list(vocabulary) + [end_token, unknown_token]     
    vocabulary_size = len(vocabulary_plus_special)     
    probabilities = {}
    
    for word in vocabulary_plus_special:
        probability = estimate_probability(word, previous_n_gram, 
                                            n_gram_counts, n_plus1_gram_counts, 
                                            vocabulary_size, k=k)
        probabilities[word] = probability
    return probabilities

def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, start_token='<s>', end_token = '</s>', k=1.0):
    """Calculates the perplexity of a given sentence using the N-gram model."""
    
    # The length of the context is determined by the size of the N-grams in the counts dictionary
    try:
        # Determine N from the keys of the N-gram (N+1 in the count variable name)
        n = len(list(n_plus1_gram_counts.keys())[0])
    except IndexError:
        # Handle case where n_plus1_gram_counts is empty
        return float('inf') 

    # Pad sentence
    # Padding is (N-1) tokens long
    sentence = [start_token] * (n-1) + sentence + [end_token]
    sentence = tuple(sentence)
    
    # N is the number of predictions made (original sentence length + 1 for end token)
    N = len(sentence) - (n - 1) 
    
    product_pi = 1.0
    
    # Start loop from the first word that requires a context (index n-1 in the padded sentence)
    for t in range(n - 1, len(sentence)): 
        # The previous context is (n-1) tokens long
        previous_n_gram = sentence[t-(n-1):t]
        word = sentence[t]

        # Use the N-gram counts for the context (previous_n_gram is N-1 tokens)
        probability = estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k)
        
        # Avoid log(0)
        if probability == 0:
            return float('inf')
            
        product_pi *= (1 / probability)
        
    # Perplexity formula: PPL = (1/P)^(1/N)
    perplexity = (product_pi)**(1/N)

    return perplexity

# --- 3. Autocomplete Functions ---

def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='</s>', unknown_token="<UNK>", k=1.0, start_with=None, n_suggestions=5):
    """
    Returns a list of top N suggestions (word, prob) tuples, correctly sorted and filtered.
    """
    
    # Determine the context size (n-1) based on the N-gram counts provided
    try:
        n = len(list(n_plus1_gram_counts.keys())[0])
    except IndexError:
        return []

    # Pad previous_tokens to ensure correct context length
    previous_tokens = ['<s>'] * (n - 1) + previous_tokens
    previous_n_gram = previous_tokens[-(n-1):] # Take the context of size n-1
    
    probabilities = estimate_probabilities(previous_n_gram,
                                             n_gram_counts, n_plus1_gram_counts,
                                             vocabulary, end_token=end_token, unknown_token=unknown_token, k=k)

    suggestions = []
    for word, prob in probabilities.items(): 
        # Filter special tokens
        if word in ('<s>', end_token, unknown_token):
             continue 
            
        # Filter by starting characters
        if start_with:
            # Check if the start_with string is a prefix of the word
            if not word.startswith(start_with): 
                 continue
                
        suggestions.append((word, prob))
        
    # Sort the suggestions by probability (descending)
    suggestions.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top N suggestions
    return suggestions[:n_suggestions]

def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None, n_suggestions=5):
    """
    Aggregates unique words from multiple N-gram models, keeps the MAX probability, 
    and returns a list of (word, probability) tuples sorted correctly.
    """
    
    # Dictionary to track the MAXIMUM probability for each unique word
    all_suggestions = {} 
    model_counts = len(n_gram_counts_list)
    
    # Iterate over N-gram models. The list is structured as [1-gram, 2-gram, 3-gram, 4-gram]
    # Starting from the highest N-gram (N=4) down to the 2-gram (N=2) model (i=1)
    # This prioritizes the most contextual models.
    for i in range(model_counts - 2, 0, -1): 
        n_gram_counts = n_gram_counts_list[i] 
        n_plus1_gram_counts = n_gram_counts_list[i+1] 
        
        # Get suggestions from this model 
        model_suggestions = suggest_a_word(
            previous_tokens, n_gram_counts, n_plus1_gram_counts, 
            vocabulary, k=k, start_with=start_with, n_suggestions=n_suggestions * 2
        )
        
        # Aggregate results: Keep the suggestion with the highest probability
        for word, prob in model_suggestions:
            if word not in all_suggestions or prob > all_suggestions[word]:
                all_suggestions[word] = prob

    # 1. Convert the unique words/probabilities from the dictionary back to a list of tuples
    final_suggestions = [(word, prob) for word, prob in all_suggestions.items()]
    
    # 2. Sort the combined list by probability (descending)
    final_suggestions.sort(key=lambda x: x[1], reverse=True)
    
    # 3. Return the top N overall suggestions
    return final_suggestions[:n_suggestions]


# --- 4. Main Execution Block (Simplified and Fixed) ---

if __name__ =="__main__":
    
    if TRAINING_MODE:
        print("--- Starting Model Training Mode ---")
        st = time.time()
        
        file_path = TRAIN_DATA_PATH # Fixed: Use the configured raw string path
        
        try:
            # Fixed: Ensure encoding="utf-8" is used for robust file reading
            with open(file_path, "r", encoding="utf-8") as f:
                train_data = f.read()
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found. Check path.")
            sys.exit(1)

        # Using a small placeholder test data for preprocessing consistency
        test_data = "The sky is clear and blue." 
        
        tokenized_train_sentences, _, vocabulary = preprocess_data(train_data, test_data, count_threshold = 2)
        et = time.time()
        print(f"Preprocessing time: {et-st:.4f}s")
        
        n_gram_counts_list = []
        # Loop up to N=4 (for 1-gram, 2-gram, 3-gram, 4-gram)
        for n in range(1, 5): 
            n_gram_counts = count_n_grams(tokenized_train_sentences, n)
            n_gram_counts_list.append(n_gram_counts)
            
        sst = time.time()
        print(f"N-gram counting time: {sst-et:.4f}s")
        
        # --- SAVE THE MODEL ---
        save_model(vocabulary, n_gram_counts_list, MODEL_FILE) 
        print(f"Total training time: {time.time() - st:.4f}s")
    
    # --- PREDICTION / DEPLOYMENT MODE (Simplified) ---
    else:
        print("--- Starting Prediction Mode (Loading Model) ---")
        st = time.time()
        
        # Load the model data
        vocabulary, n_gram_counts_list = load_model(MODEL_FILE)
        
        if not vocabulary or not n_gram_counts_list:
            print("Cannot run in prediction mode without a trained model file. Please set TRAINING_MODE = True and run once.")
            sys.exit(1)
            
        et = time.time()
        print(f"Model loading time: {et-st:.4f}s")
        print(f"Vocab size: {len(vocabulary)}")

        
        # --- FIXED PREDICTION EXAMPLE ---
        
        # 1. Define context tokens (User Input)
        user_input_string = input(" \n Type a few words: \n")
        
        # Tokenize the user input string into a list of words (this returns a list of lists of tokens)
        tokenized_input_sentences = tokenize_data(user_input_string)
        
        if not tokenized_input_sentences:
            print("Input is empty or resulted in no tokens.")
            sys.exit(0)
            
        context_words = tokenized_input_sentences[0] # Get the first (and only) sentence token list

        # 2. Process the tokens to replace OOV words with <UNK>
        # Note: replace_oov_words_by_unk expects a list of tokenized sentences, 
        # so we pass it [context_words] and extract the first (and only) sentence back out.
        input_tokens_processed = replace_oov_words_by_unk([context_words], vocabulary)[0]
        
        print(f"\nEvaluating sequence and predicting next word after context: {' '.join(context_words)}")
        
        # --- PERPLEXITY CALCULATION ---
        # To use the 3-gram model (N=3):
        # We need N-gram (context/denominator) counts: 2-gram (Index 1)
        # We need (N+1)-gram (numerator) counts: 3-gram (Index 2)
        
        # Index 1 holds 2-gram counts (N-1 for 3-gram model)
        n_minus1_counts = n_gram_counts_list[1] 
        # Index 2 holds 3-gram counts (N for 3-gram model)
        n_counts = n_gram_counts_list[2] 
        # Vocabulary size must include special tokens (</s> and <UNK>)
        vocab_size_for_ppl = len(vocabulary) + 2

        # Calculate Perplexity for the input sequence
        perplexity = calculate_perplexity(
            input_tokens_processed, 
            n_minus1_counts, 
            n_counts, 
            vocab_size_for_ppl
        )
        
        print(f"\n--- Perplexity Score (using 3-gram model) ---")
        print(f"Sequence Perplexity: {perplexity:.4f}")
        
        # --- SUGGESTIONS CALCULATION ---
        sst = time.time()
        # Predict the next word (no prefix is being typed in this simplified example)
        suggestions = get_suggestions(input_tokens_processed, n_gram_counts_list, vocabulary, k=1.0, start_with=None)
        tt = time.time()
        
        print(f"Time taken for suggestions: {tt-sst:.4f}s")
        
        print("\n--- Top Autocomplete Suggestions (Word, Max Probability) ---")
        if suggestions:
            for word, prob in suggestions:
                print(f"'{word}' (P={prob:.10f})")
        else:
            print("No suggestions found.")
