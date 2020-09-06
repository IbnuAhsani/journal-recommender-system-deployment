# Journal Recommender System Website

A recommender system that recommends Indonesian scientific journals based on the article abstracts that is used as input. A deployed version is available through [this](https://sistem-rekomendasi-jurnal.herokuapp.com/) link. 

## Features

- Utilizes a trained [Softmax Regression](https://github.com/ssentinull/softmax-regression-module) model and [Chi-Square](https://github.com/ssentinull/chi-square-module) feature selection.
- Outputs journal recommendations in the form of probabilities.
- Recommends journal with the highest probability as the main recommendation, and outputs a summary and a link for said recommended journal.
- Works properly only with **Indonesian** abstracts.
- Only allows abstracts that consist of 100 - 350 words as input.
- Recommends 12 different journals:

| No. | Name | Scope of Knowledge |
| -------- | ---------- | -------- 
| 1. | [Jurnal Hortikultura](http://ejurnal.litbang.pertanian.go.id/index.php/jhort) | Horticulture |
| 2. | [Jurnal Penelitian Perikanan Indonesia](http://ejournal-balitbang.kkp.go.id/index.php/jppi) | Fisheries | 
| 3. | [Jurnal Riset Akuakultur](http://ejournal-balitbang.kkp.go.id/index.php/jra) | Aquaculture |
| 4. | [Jurnal Jalan-Jembatan](http://jurnal.pusjatan.pu.go.id/index.php/jurnaljalanjembatan) | Road Construction |
| 5. | [Jurnal Penelitian Hasil Hutan](http://ejournal.forda-mof.org/ejournal-litbang/index.php/JPHH/index) | Forestries | 
| 6. | [Jurnal Penelitian Hutan dan Konservasi Alam](http://ejournal.forda-mof.org/ejournal-litbang/index.php/JPHKA) | Forest Conservation | 
| 7. | [E-Jurnal Medika Udayana](https://ojs.unud.ac.id/index.php/eum) | Medical Sciences | 
| 8. | [Jurnal Simetris](https://jurnal.umk.ac.id/index.php/simet) | Technology |
| 9. | [Jurnal Teknik ITS](http://ejurnal.its.ac.id/index.php/teknik) | Technology |
| 10. | [Berita Kedokteran Masyarakat](https://jurnal.ugm.ac.id/bkm) | Public Health |
| 12. | [Indonesia Medicus Veterinus](https://ojs.unud.ac.id/index.php/imv/index) | Veterinary |
| 13. | [Matriks Teknik Sipil](https://jurnal.uns.ac.id/matriks) | Civil Engineering | 
<br/>

## How to Configure in Local Environment

After cloning the repo, do the following steps:

1. create a virtual environment in the cloned dir

    ````shell
    $ python3 -m venv venv
    ````

2. activate the virtual environment

    ````shell
    $ source venv/bin/activate
    ````

3. install all the dependencies listed in `requirements.txt`

    ````shell
    $ pip install -r requirements.txt
    ````

4. setup the flask environment variable in `.env`

    ````env
    SECRET_KEY=your_secret_key
    ABSTRACT_TOKEN_SAVE_DIR=./data/output/abstract-token-list.json
    TF_IDF_SAVE_DIR=./data/output/tf-idf.csv
    FV_TOKENS_OPEN_DIR=./data/fv-tokens
    JOURNAL_DATA_OPEN_DIR=./static/journal_info
    ````

5. deactivate and reactivate the virtual environment

    ````shell
    $ deactivate
    $ source venv/bin/activate
    ````

6. export the shell environment variables

    ````shell
    $ export FLASK_APP=server.py
    $ export FLASK_ENV=development
    ````

7. run the app

    ````shell
    $ flask run
    ````

## How to Run in Local Environment

After successfully configuring the program, do the following steps everytime you want to run the app:

1. activate the virtual environment

    ````shell
    $ source venv/bin/activate
    ````

2. export the shell environment variables

    ````shell
    $ export FLASK_APP=server.py
    $ export FLASK_ENV=development
    ````

3. run the app

    ````shell
    $ flask run
    ````

## Library Used

- [Flask](https://flask.palletsprojects.com/en/1.1.x/)
- [Flask-Bootstrap](https://pythonhosted.org/Flask-Bootstrap/)
- [Jinja](https://jinja.palletsprojects.com/en/2.11.x/)
- [NumPy](https://numpy.org/)
- [Sastrawi](https://pypi.org/project/Sastrawi/)
- [WTForms](https://wtforms.readthedocs.io/en/2.3.x/)

## Demo

- Input validation

![](https://media.giphy.com/media/J3SLW8RvR55zMea4h1/giphy.gif)

- Recommendation result

![](https://media.giphy.com/media/daJ6Z7uG5e8Four7Mj/giphy.gif)

## License

MIT © [ssentinull](https://github.com/ssentinull)