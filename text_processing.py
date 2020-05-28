import csv
import json
import math
import nltk
import string
import numpy as np
from collections import OrderedDict 
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


def test_append(text):
    result = " ".join(('hello', text, ", my name is ibnu"))
    
    return result


def preprocess(abstract):

    stop_word_factory = StopWordRemoverFactory()
    stemmer_factory = StemmerFactory()
    stopword = stop_word_factory.create_stop_word_remover()
    stemmer = stemmer_factory.create_stemmer()

    abstract_lower_case = abstract.lower()
    abstract_white_spaces_removed = abstract_lower_case.strip()
    abstract_numbers_removed = abstract_white_spaces_removed.translate(string.digits)
    abstract_punctuation_removed = abstract_numbers_removed.translate(string.punctuation)
    abstract_stopword_removed = stopword.remove(
        abstract_punctuation_removed)
    abstract_stemmed = stemmer.stem(abstract_stopword_removed)
    abstract_tokens = nltk.word_tokenize(abstract_stemmed)

    final_token = []

    for token in abstract_tokens:
      token_digits_removed = ''.join([i for i in token if not i.isdigit()])
      final_token.append(token_digits_removed)

    return final_token


def calculate_tf(word_dict, token_list):
    tf_dict = {}
    token_list_length = len(token_list)

    for word, value in word_dict.items():
        word_count = token_list.count(word)
        tf = word_count / token_list_length
        tf_dict[word] = tf 
    
    return tf_dict


def calculate_idf(documents):
    num_documents = len(documents)
    idf_dict = dict.fromkeys(documents[0].keys(), 0)

    for document in documents:
        for word, val in document.items():
            if val > 0:
                idf_dict[word] += 1
    
    for word, val in idf_dict.items():
        idf_dict[word] = 1 + math.log(num_documents / float(val + 1))
    
    return idf_dict


def calculate_tf_idf(tf, idf):
    tf_idf = {}

    for word, val in tf.items():
        if val not in idf:
            word_idf = 1
        else:
            word_idf = idf[word]

        tf_idf[word] = val * word_idf
            
    return tf_idf


def predict(abstract, model):
    abstract_token_list = preprocess(abstract)

    with open('./data/abstract-token-list.json', 'w') as f:
        json.dump(abstract_token_list , f, indent=4)

    with open('./data/fv-tokens.json') as f:
      fv_token_list = json.load(f)
    
    fv_token_dict = OrderedDict({ i : 0 for i in fv_token_list})
    tfs = calculate_tf(fv_token_dict, abstract_token_list)

    tf_list = []

    for tf in tfs.values():
      tf_list.append(tf) 

    with open('./data/tf-idf.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        wr.writerow(tf_list)

    tf_list_np = np.array([tf_list])

    probabilities = model.predict_proba(tf_list_np)
    print("probabilities: ", probabilities)

    predict = model.predict(tf_list_np)
    print("predict: ", predict)

    return predict