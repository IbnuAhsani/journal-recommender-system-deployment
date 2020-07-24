import os
import pickle
import sklearn
import numpy as np
from operator import itemgetter
from SubmitForm import SubmitForm
from utils import text_processing
from flask import Flask, jsonify, render_template
from flask_bootstrap import Bootstrap

STATIC_PIC_DIR = os.path.join('static', 'journal_cover')

app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', None)
app.config['STATIC_PIC_DIR'] = STATIC_PIC_DIR

Bootstrap(app)

model = pickle.load(open('./classifiers/final_model_data_23_150_feature.pkl', 'rb'))

@app.route("/", methods=['GET', 'POST'])
def home():
  form = SubmitForm()

  if form.validate_on_submit():
    abstract = form.abstract.data
    num_abstract_words = len(abstract.split())

    if num_abstract_words < 100:
        word_count_validation_message = 'Abstrak Anda harus lebih dari 100 kata'
        return render_template("home.html", title='Journal Recommender System', 
            form=form, word_count_validation_message=word_count_validation_message)

    if num_abstract_words > 350:
        word_count_validation_message = 'Abstrak Anda tidak boleh lebih dari 350 kata'
        return render_template("home.html", title='Journal Recommender System', 
            form=form, word_count_validation_message=word_count_validation_message)

    prediction, probabilities = text_processing.predict(abstract, model)
    journal_cover_name = prediction['JOURNAL_COVER']
    journal_cover_path = os.path.join(app.config['STATIC_PIC_DIR'], journal_cover_name)
    prediction['JOURNAL_COVER'] = journal_cover_path

    probabilities_sorted_desc = sorted(probabilities, key=itemgetter('JOURNAL_PROBABILITY'), reverse=True)
    
    return render_template("home.html", title='Journal Recommender System', 
        form=form, prediction=prediction, probabilities=probabilities_sorted_desc)

  return render_template("home.html", title='Journal Recommender System', form=form)

