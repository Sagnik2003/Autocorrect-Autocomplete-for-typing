import math
import random
import numpy as np
import pandas as pd
import nltk
# Assuming the user has run nltk.download('punkt') successfully 
# in the environment's terminal/interpreter as needed.

from nltk.tokenize import word_tokenize
# Setting nltk.data.path.append('.') is generally fine, 
# but often the source of LookupErrors if the data isn't there.
# Use absolute paths if errors persist!
# nltk.data.path.append('.') 

with open("Autocorrect and Autocomplete/App/data/shakespeare.txt", "r") as f:
    data = f.read()
    
def split_to_sentences(data):
    # Use splitlines(True) if you want to keep the newline, but split("\n") is fine here
    sentences = data.split("\n") 
    
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]
    return sentences

def tokenize_sentences(sentences):
    # Renamed the internal list variable to avoid conflict with the function name
    tokenized_list = []
    for sentence in sentences:
        sentence = sentence.lower()
        # Use nltk.word_tokenize on the single sentence string
        tokenized = word_tokenize(sentence)
        tokenized_list.append(tokenized)
    return tokenized_list

# # test your code
# sentences = ["Sky is blue.", "Leaves are green.", "Roses are red."]
# print(tokenize_sentences(sentences))

def tokenize_data(data):
    sentences = split_to_sentences(data)
    tokenized_sentences = tokenize_sentences(sentences)
    return tokenized_sentences

def count_words(tokenized_sentences):
    word_counts = {}
    for sentence in tokenized_sentences:
        for token in sentence:
            if token not in word_counts.keys():
                word_counts[token] = 1
            else:
                word_counts[token] += 1
    return word_counts

def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    word_counts = count_words(tokenized_sentences)
    closed_vocab = [word for word, count in word_counts.items() if count >= count_threshold]
    return closed_vocab

def replace_oov_words_by_unk(tokenized_sentences, vocabulary,unknown_token="<UNK>"):
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
    tokenized_train_sentences = tokenize_data(train_data)
    vocabulary = get_words_with_nplus_frequency_fn(tokenized_train_sentences, count_threshold)
    
    tokenized_train_sentences = replace_oov_words_by_unk(tokenized_train_sentences, vocabulary, unknown_token)
    
    tokenized_test_sentences = tokenize_data(test_data)
    tokenized_test_sentences = replace_oov_words_by_unk(tokenized_test_sentences, vocabulary, unknown_token)
    
    return tokenized_train_sentences, tokenized_test_sentences, vocabulary


def count_n_grams(data, n , start_token = '<s>', end_token = '</s>'):
    n_gram_counts = {}
    for sentence in data:
        # Add start and end tokens
        sentence = [start_token] * n + sentence + [end_token]
        L  = len(sentence)
        m = L - n + 1
        for i in range(m):
            n_gram = tuple(sentence[i:i+n])
            if n_gram in n_gram_counts.keys():
                n_gram_counts[n_gram] += 1
            else:
                n_gram_counts[n_gram] = 1
    return n_gram_counts

def estimate_probability(word,previous_n_gram,n_gram_counts, n_plus1_gram_counts, vocabulary_size, k = 1.0):
    previous_n_gram = tuple(previous_n_gram)
    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)
    denominator = previous_n_gram_count + k * vocabulary_size
    n_plus1_gram = previous_n_gram + (word,)
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)
    numerator = n_plus1_gram_count + k
    probability = numerator / denominator
    return probability

def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='<e>', unknown_token="<unk>",  k=1.0):

    previous_n_gram = tuple(previous_n_gram)    
    vocabulary = vocabulary + [end_token, unknown_token]    
    vocabulary_size = len(vocabulary)    
    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram, 
                                           n_gram_counts, n_plus1_gram_counts, 
                                           vocabulary_size, k=k)
        probabilities[word] = probability
    return probabilities

def make_count_matrix(n_plus1_gram_counts, vocabulary):
    # add <e> <unk> to the vocabulary
    # <s> is omitted since it should not appear as the next word
    vocabulary = vocabulary + ["<e>", "<unk>"]
    
    # obtain unique n-grams
    n_grams = []
    for n_plus1_gram in n_plus1_gram_counts.keys():
        n_gram = n_plus1_gram[0:-1]        
        n_grams.append(n_gram)
    n_grams = list(set(n_grams))
    
    # mapping from n-gram to row
    row_index = {n_gram:i for i, n_gram in enumerate(n_grams)}    
    # mapping from next word to column
    col_index = {word:j for j, word in enumerate(vocabulary)}    
    
    nrow = len(n_grams)
    ncol = len(vocabulary)
    count_matrix = np.zeros((nrow, ncol))
    for n_plus1_gram, count in n_plus1_gram_counts.items():
        n_gram = n_plus1_gram[0:-1]
        word = n_plus1_gram[-1]
        if word not in vocabulary:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i, j] = count
    
    count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=vocabulary)
    return count_matrix

def make_probability_matrix(n_plus1_gram_counts, vocabulary, k):
    count_matrix = make_count_matrix(n_plus1_gram_counts, vocabulary)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return prob_matrix


def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, start_token='<s>', end_token = '<e>', k=1.0):
    n = len(list(n_gram_counts.keys())[0]) 
    sentence = [start_token] * n + sentence + [end_token]
    sentence = tuple(sentence)
    N = len(sentence)
    product_pi = 1.0
    for t in range(n, N):
        n_gram = sentence[t-n:t]
        word = sentence[t]
        probability = estimate_probability(word, n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k)
        product_pi *= 1/probability
    perplexity = (product_pi)**(1/N)

    return perplexity

# UNQ_C11 GRADED FUNCTION: suggest_a_word
def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='<e>', unknown_token="<unk>", k=1.0, start_with=None):
    """
    Get suggestion for the next word
    
    Args:
        previous_tokens: The sentence you input where each token is a word. Must have length >= n 
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: positive constant, smoothing parameter
        start_with: If not None, specifies the first few letters of the next word
        
    Returns:
        A tuple of 
          - string of the most likely next word
          - corresponding probability
    """
    
    # length of previous words
    n = len(list(n_gram_counts.keys())[0])
    
    # append "start token" on "previous_tokens"
    previous_tokens = ['<s>'] * n + previous_tokens
    
    # From the words that the user already typed
    # get the most recent 'n' words as the previous n-gram
    previous_n_gram = previous_tokens[-n:]

    # Estimate the probabilities that each word in the vocabulary
    # is the next word,
    # given the previous n-gram, the dictionary of n-gram counts,
    # the dictionary of n plus 1 gram counts, and the smoothing constant
    probabilities = estimate_probabilities(previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary, k=k)
    
    # Initialize suggested word to None
    # This will be set to the word with highest probability
    suggestion = None
    
    # Initialize the highest word probability to 0
    # this will be set to the highest probability 
    # of all words to be suggested
    max_prob = 0
    
    ### START CODE HERE ###
    
    # For each word and its probability in the probabilities dictionary:
    for word, prob in probabilities.items(): # complete this line
        
        # If the optional start_with string is set
        if start_with: # complete this line with the proper condition
            
            # Check if the beginning of word does not match with the letters in 'start_with'
            if not word.startswith(start_with): # complete this line with the proper condition

                # if they don't match, skip this word (move onto the next word)
                continue
        
        # Check if this word's probability
        # is greater than the current maximum probability
        if prob > max_prob: # complete this line with the proper condition
            
            # If so, save this word as the best suggestion (so far)
            suggestion = word 
            
            # Save the new maximum probability
            max_prob = prob

    ### END CODE HERE
    
    return suggestion, max_prob


# # test your code
# sentences = [['i', 'like', 'a', 'cat'],
#              ['this', 'dog', 'is', 'like', 'a', 'cat']]
# unique_words = list(set(sentences[0] + sentences[1]))

# unigram_counts = count_n_grams(sentences, 1)
# bigram_counts = count_n_grams(sentences, 2)

# previous_tokens = ["i", "like"]
# tmp_suggest1 = suggest_a_word(previous_tokens, unigram_counts, bigram_counts, unique_words, k=1.0)
# print(f"The previous words are 'i like',\n\tand the suggested word is `{tmp_suggest1[0]}` with a probability of {tmp_suggest1[1]:.4f}")

# print()
# # test your code when setting the starts_with
# tmp_starts_with = 'c'
# tmp_suggest2 = suggest_a_word(previous_tokens, unigram_counts, bigram_counts, unique_words, k=1.0, start_with=tmp_starts_with)
# print(f"The previous words are 'i like', the suggestion must start with `{tmp_starts_with}`\n\tand the suggested word is `{tmp_suggest2[0]}` with a probability of {tmp_suggest2[1]:.4f}")

def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts-1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i+1]
        
        suggestion = suggest_a_word(previous_tokens, n_gram_counts,
                                    n_plus1_gram_counts, vocabulary,
                                    k=k, start_with=start_with)
        suggestions.append(suggestion)
    return suggestions

