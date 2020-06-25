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


def calculate_tf_idf(tf):
    tf_idf_dict = {}

    for word, val in tf.items():
        if val == 0:
            tf_idf = val
        else:
            tf_idf = val * math.log10(1/2)

        tf_idf_dict[word] = tf_idf
            
    return tf_idf_dict


def predict(abstract, model):

    ABSTRACT_TOKEN_SAVE_DIR = './data/abstract-token-list.json'
    TF_IDF_SAVE_DIR = './data/tf-idf.csv'
    FV_TOKENS_OPEN_DIR = './data/fv-tokens.json'
    JOURNAL_DATA_OPEN_DIR = './static/journal_info/journal-info.json'

    abstract_token_list = preprocess(abstract)

    with open(ABSTRACT_TOKEN_SAVE_DIR, 'w') as f:
        json.dump(abstract_token_list , f, indent=4)

    with open(FV_TOKENS_OPEN_DIR) as f:
      fv_token_list = json.load(f)
    
    fv_token_dict = OrderedDict({ i : 0 for i in fv_token_list})
    tfs = calculate_tf(fv_token_dict, abstract_token_list)
    tfidfs = calculate_tf_idf(tfs)

    tf_idf_list = []

    for tfidf in tfidfs.values():
      tf_idf_list.append(tfidf) 

    with open(TF_IDF_SAVE_DIR, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        wr.writerow(tf_idf_list)

    tf_idf_list_np = np.array([tf_idf_list])

    probabilities = model.predict_proba(tf_idf_list_np)
    print("probabilities: ", probabilities)

    predict = model.predict(tf_idf_list_np)
    print("predict: ", predict)

    with open(JOURNAL_DATA_OPEN_DIR) as f:
      journal_datas = json.load(f)

    journal_data = journal_datas[str(predict[0])]

    return journal_data