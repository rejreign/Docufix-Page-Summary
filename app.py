from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
from flask import Flask, request, jsonify, render_template, redirect

# from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk

LANGUAGE = "english"
SENTENCES_COUNT = 10

app = Flask(__name__)

@app.route('/', methods=['GET'])
def homepage():
    return render_template('paraphrase.html')


@app.route('/', methods=['POST'])
def summarize():
    """ Returns summary of articles """
    text = request.form['text']
    # parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    parser = PlaintextParser.from_string(text,Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    final = []

    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        final.append(str(sentence))
    length = len(final)
    return render_template('paraphrase.html',report=final,length=length)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)