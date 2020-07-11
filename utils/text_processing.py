import csv
import json
import math
import nltk
import string
import numpy as np
from collections import OrderedDict 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


def test_append(text):
    result = " ".join(('hello', text, ", my name is ibnu"))
    
    return result


def preprocess(abstract):

    with open('./data/stop-words.json') as f:
        stop_words = json.load(f)

    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()

    remove_digits = str.maketrans(string.digits, ' '*len(string.digits))
    remove_punctuations = str.maketrans(string.punctuation, ' '*len(string.punctuation))

    abstract_lower_case = abstract.lower()
    abstract_white_spaces_removed = abstract_lower_case.strip()
    abstract_numbers_removed =  abstract_white_spaces_removed.translate(remove_digits)
    abstract_punctuation_removed = abstract_numbers_removed.translate(remove_punctuations)
    abstract_punctuation_removed_list = abstract_punctuation_removed.split()
    abstract_sw_removed_list = [word for word in abstract_punctuation_removed_list if word not in stop_words]
    abstract_sw_removed = " ".join(abstract_sw_removed_list)    
    abstract_stemmed = stemmer.stem(abstract_sw_removed)
    abstract_tokens = nltk.word_tokenize(abstract_stemmed)

    tokens_digits_removed = []

    for token in abstract_tokens:
      token_digits_removed = ''.join([i for i in token if not i.isdigit()])
      tokens_digits_removed.append(token_digits_removed)
    
    final_token = []

    for token in tokens_digits_removed:
        if token not in final_token:
            final_token.append(token)

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

    ABSTRACT_TOKEN_SAVE_DIR = './data/output/abstract-token-list.json'
    TF_IDF_SAVE_DIR = './data/output/tf-idf.csv'
    FV_TOKENS_OPEN_DIR = './data/fv-tokens/final-fv-tokens-data-23-150-feature.json'
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
    probabilities_data = []
    index = 0

    for probability in probabilities[0]:
        temp_dict = {
            'JOURNAL_ID': index,
            'JOURNAL_NAME': journal_datas[str(index)]['JOURNAL_NAME'],
            'JOURNAL_PROBABILITY': round(probability * 100, 2),
        }

        probabilities_data.append(temp_dict)
        index += 1

    return journal_data, probabilities_data