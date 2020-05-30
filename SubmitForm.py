from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.widgets import TextArea
from wtforms.validators import DataRequired


class SubmitForm(FlaskForm):
  abstract = StringField('Ketik Abstrak Jurnal Anda', widget=TextArea(), validators=[DataRequired()])
  submit = SubmitField('submit')