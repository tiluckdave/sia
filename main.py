from flask import Flask, request
from threading import Thread
from flask_cors import cross_origin
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)


@app.route('/sia', methods=['POST'])
@cross_origin()
def sia():
  sia = SentimentIntensityAnalyzer()
  sentence = request.json['sentence']
  data = sia.polarity_scores(sentence)
  return data


def run():
  app.run(host='0.0.0.0', port=8080)


t = Thread(target=run)
t.start()
