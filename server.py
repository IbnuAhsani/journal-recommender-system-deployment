import pickle
import sklearn
import numpy as np
from SubmitForm import SubmitForm
from utils import text_processing
from flask import Flask, jsonify, render_template

app = Flask(__name__)
app.config['SECRET_KEY'] = '123456789'

model = pickle.load(open('./classifiers/model_sastrawi.pkl', 'rb'))

@app.route("/", methods=['GET', 'POST'])
def home():
  form = SubmitForm()

  if form.validate_on_submit():
    abstract = form.abstract.data
    prediction = text_processing.predict(abstract, model)

    return '<h1> the the class is {}.'. format(prediction)

  return render_template("home.html", title='Form Submition', form=form)

