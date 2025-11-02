import re
from collections import Counter
import numpy as np
import pandas as pd
import os

# file_path = 'Autocorrect and Autocomplete/App/data/shakespeare.txt'
# print("Looking for:", os.path.abspath(file_path))
# print("Exists:", os.path.exists(file_path))


def process_data(file_name):
    words = []
    with open(file_name, 'r',encoding='utf-8') as file:
        file_content = file.read()
    file_content = file_content.lower()
    
    words = re.findall(r'\w+', file_content )
    return words


# print(f"Total unique words: {len(vocab)}")

def get_count(word_l):
    word_count_dict = {}
    
    for word in set(word_l):
        word_count_dict[word] = word_l.count(word)
        
    return word_count_dict


# word_count_dict = get_count(word_l)
# print(f"There are {len(word_count_dict)} key values pairs")
# print(f"The count for the word 'thee' is {word_count_dict.get('thee',0)}")
    
def get_probs(word_count_dict):
    probs = {}
    
    M = sum(word_count_dict.values())
    for word in word_count_dict.keys():
        probs[word] = word_count_dict[word] / M
    return probs

# probs = get_probs(word_count_dict)
# print(f"Length of probs is {len(probs)}")
# print(f"P('thee') is {probs['thee']:.4f}")

def delete_letter(word,verbose = False):
    delete_l = []
    split_l = []
    
    split_l = [(word[:i], word[i:]) for i in range(len(word)+1)]
    delete_l = [(a+b[1:])for a,b in split_l]
    
    if verbose:
        print(f"delete_letter('{word}')")
        print(f"split_l: {split_l}")
        print(f"delete_l: {delete_l}")
    return delete_l

# delete_word_l = delete_letter(word="cans",
#                         verbose=True)

def switch_leter(word,verbose = False):
    switch_l = []
    split_l = []
    
    split_l = [(word[:i], word[i:]) for i in range(len(word)-1)]
    switch_l = [a + b[1] + b[0] + b[2:] for a,b in split_l]
    
    if verbose:
        print(f"switch_letter('{word}')")
        print(f"split_l: {split_l}")
        print(f"switch_l: {switch_l}")
    return switch_l

# switch_word_l = switch_leter(word="cans",
#                         verbose=True)

def replace_letter(word,verbose = False):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    
    replace_l = []
    split_l = []
    
    split_l = [(word[:i], word[i:]) for i in range(len(word))]
    replace_l = [a + l +(b[1:] if len(b)>1 else '') for a,b in split_l if b for l in letters]
    replace_l = set(replace_l)
    replace_l.remove(word)
    
    replace_l = sorted(replace_l)
    
    if verbose:
        print(f"replace_letter('{word}')")
        print(f"split_l: {split_l}")
        print(f"replace_l: {replace_l}")
    return replace_l

# replace_word_l = replace_letter(word="cans",
#                         verbose=True)

def insert_letter(word,verbose = False):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    
    insert_l = []
    split_l = []
    
    split_l = [(word[:i], word[i:]) for i in range(len(word)+1)]
    insert_l = [a + l + b for a,b in split_l for l in letters]
    
    if verbose:
        print(f"insert_letter('{word}')")
        print(f"split_l: {split_l}")
        print(f"insert_l: {insert_l}")
    return insert_l

# insert_l = insert_letter('at', True)
# print(f"Number of strings output by insert_letter('at') is {len(insert_l)}")

def edit_one_letter(word,allow_switches = True):
    edit_one_set = set()
    edit_one_set.update(delete_letter(word))
    edit_one_set.update(replace_letter(word))
    edit_one_set.update(insert_letter(word))
    if allow_switches:
        edit_one_set.update(switch_leter(word))
        
    return edit_one_set


# tmp_word = "at"
# tmp_edit_one_set = edit_one_letter(tmp_word)
# # turn this into a list to sort it, in order to view it
# tmp_edit_one_l = sorted(list(tmp_edit_one_set))

# print(f"input word {tmp_word} \n edit_one_l \n{tmp_edit_one_l}\n")
# print(f"The type of the returned object should be a set {type(tmp_edit_one_set)}")
# print(f"Number of outputs from edit_one_letter('at') is {len(edit_one_letter('at'))}")

def edit_two_letters(word, allow_switches = True):
    edit_two_set = set()
    edit_one = edit_one_letter(word,allow_switches = allow_switches)
    for w in edit_one:
        if w:
            edit_two = edit_one_letter(w,allow_switches= allow_switches)
            edit_two_set.update(edit_two)
    return set(edit_two_set)


# tmp_edit_two_set = edit_two_letters("a")
# tmp_edit_two_l = sorted(list(tmp_edit_two_set))
# print(f"Number of strings with edit distance of two: {len(tmp_edit_two_l)}")
# print(f"First 10 strings {tmp_edit_two_l[:10]}")
# print(f"Last 10 strings {tmp_edit_two_l[-10:]}")
# print(f"The data type of the returned object should be a set {type(tmp_edit_two_set)}")
# print(f"Number of strings that are 2 edit distances from 'at' is {len(edit_two_letters('at'))}")

# def edit_three_letters(word, allow_switches = True):
#     edit_three_set = set()
#     edit_two = edit_two_letters(word,allow_switches = allow_switches)
#     for w in edit_two:
#         if w:
#             edit_three = edit_one_letter(w,allow_switches= allow_switches)
#             edit_three_set.update(edit_three)
#     return set(edit_three_set)


def get_corrections(word, probs,vocab, n=3, verbose = True):
    suggestions = []
    n_best = []
    
    suggestions = list((word in vocab and word)or edit_one_letter(word).intersection(vocab) or edit_two_letters(word).intersection(vocab)) # or edit_three_letters(word).intersection(vocab))
    n_best = [[s,probs.get(s,0)] for s in list(reversed(suggestions))]
    
    if verbose:
        print("entered word:", word)
        print("suggestions:", suggestions)
    return n_best

# # Test your implementation - feel free to try other words in my word
# my_word = 'dys' 
# tmp_corrections = get_corrections(my_word, probs, vocab,n=3, verbose=True) # keep verbose=True
# for i, word_prob in enumerate(tmp_corrections):
#     print(f"word {i}: {word_prob[0]}, probability {word_prob[1]:.6f}")

# # CODE REVIEW COMMENT: using "tmp_corrections" insteads of "cors". "cors" is not defined
# print(f"data type of corrections {type(tmp_corrections)}")

def min_edit_distance(source,target, ins_cost = 1, del_cost = 1, rep_cost = 2):
    m = len(source)
    n = len(target)
    
    D = np.zeros((m+1,n+1), dtype=int)
    
    for row in range(1,m+1):
        D[row,0] = D[row-1,0] + del_cost
        
    for col in range(1,n+1):
        D[0,col] = D[0,col-1] + ins_cost
    
    for row in range(1,m+1):
        for col in range(1,n+1):
            r_cost =  rep_cost
            if source[row-1] == target[col-1]:
                r_cost = 0
            D[row,col] = min(D[row-1,col] + del_cost,
                            D[row,col-1] + ins_cost,
                            D[row-1,col-1] + r_cost)
    med = D[m,n]
    return D, med


# source =  'play'
# target = 'stay'
# matrix, min_edits = min_edit_distance(source, target)
# print("minimum edits: ",min_edits, "\n")
# idx = list('#' + source)
# cols = list('#' + target)
# df = pd.DataFrame(matrix, index=idx, columns= cols)
# print(df)


# #DO NOT MODIFY THIS CELL
# # testing your implementation 
# source =  'eer'
# target = 'near'
# matrix, min_edits = min_edit_distance(source, target)
# print("minimum edits: ",min_edits, "\n")
# idx = list(source)
# idx.insert(0, '#')
# cols = list(target)
# cols.insert(0, '#')
# df = pd.DataFrame(matrix, index=idx, columns= cols)
# print(df)
def display_med_matrix(source,target,matrix):
    idx = list(source)
    idx.insert(0, '#')
    cols = list(target)
    cols.insert(0, '#')
    df = pd.DataFrame(matrix, index=idx, columns= cols)
    print("\n")
    print(df)
    print("\n")
    
    
def get_corrections_by_med(word, probs,vocab, n=3, verbose = True, display_matrix = False):
    suggestions = []
    n_best = []
    
    suggestions = list((word in vocab and word)or edit_one_letter(word).intersection(vocab) or edit_two_letters(word).intersection(vocab) ) # or edit_three_letters(word).intersection(vocab))
    
    med_list = []
    for s in suggestions:
        D, med = min_edit_distance(word, s)
        med_list.append((s, med, probs.get(s,0)))
    
    # sort by min edit distance first, then by probability
    med_list = sorted(med_list, key=lambda x: (x[1], -x[2]))
    
    n_best = med_list[:n]
    
    autocorrected_words = [w[0] for w in n_best]
    if verbose:
        print("entered word:", word)
        print("suggestions:", autocorrected_words)
    if display_matrix:
        for w in autocorrected_words:
            D, _ = min_edit_distance(word, w)
            display_med_matrix(word, w, D)
    return autocorrected_words

# get_corrections_by_med('pyy', probs, vocab, n=3, verbose=True,display_matrix=False)

if __name__ == "__main__":
    word_l = process_data('Autocorrect and Autocomplete/App/data/AllCombined.txt') #Autocorrect and Autocomplete\App\data\demo.txt
    vocab = set(word_l)
    probs = get_probs(get_count(word_l))
    # Example usage
    word = input("Enter a word for autocorrection: ")
    # Get corrections using the new method
    corrections = get_corrections_by_med(word, probs, vocab, n=3, verbose=True, display_matrix=False)
    print(f"Corrections for '{word}': {corrections}")