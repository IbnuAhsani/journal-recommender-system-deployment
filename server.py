import pickle
import sklearn
import numpy as np
from SubmitForm import SubmitForm
from text_processing import predict
from flask import Flask, jsonify, render_template

app = Flask(__name__)
app.config['SECRET_KEY'] = '123456789'

model = pickle.load(open('model.pkl', 'rb'))

@app.route("/", methods=['GET', 'POST'])
def home():
  form = SubmitForm()

  if form.validate_on_submit():
    abstract = form.abstract.data
    prediction = predict(abstract)

    return '<h1> the username is {}.'. format(prediction)

  return render_template("home.html", title='Form Submition', form=form)

