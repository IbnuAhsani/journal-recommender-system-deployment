from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class SubmitForm(FlaskForm):
  abstract = StringField('abstract', validators=[DataRequired()])
  submit = SubmitField('submit')