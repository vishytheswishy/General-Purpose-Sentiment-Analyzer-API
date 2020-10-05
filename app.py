import pandas as pd
import pickle
import json
import re
import emoji
import nltk
#--------------------------------------------------------------------------------#
from flask import Flask, request, render_template
from flask_restful import Api, Resource
from textblob import TextBlob
#--------------------------------------------------------------------------------#
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#--------------------------------------------------------------------------------#
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
stopwords.remove("not")
stopwords.remove("no")
stopwords.remove("nor")
stopwords.remove("above")
#--------------------------------------------------------------------------------#
lemma = WordNetLemmatizer()

# Do this first, that'll do something eval() 
# to "materialize" the LazyCorpusLoader
next(swn.all_senti_synsets()) 
#--------------------------------------------------------------------------------#

pattern = '@\S+|https?:\S+|http?:\S|[^A-Za-z]+|com|net'
urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
userPattern       = '@[^\s]+'
alphaPattern      = "[^a-zA-Z0-9]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

#--------------------------------------------------------------------------------#
file = open('LR.pkl', 'rb')
LRmodel = pickle.load(file)
file.close()

file = open('vectoriser.pkl','rb')
vectoriser = pickle.load(file)
file.close()

app = Flask(__name__, template_folder='template')
api = Api(app)
#--------------------------------------------------------------------------------#
"""
Helper Functions: 
def predict: (make predictions on text and return json)
def tokenize: tokenze and parse texts (remove emoji's and stopwords)
def determine_sentiment_using_blob: implemented polarity to normalize models (neutral)
"""
#--------------------------------------------------------------------------------#
def predict(vectoriser, model, text):

    # Predict the sentiment
    listD = tokenize(str(text).lower())
    listD = [listD]
    textdata = vectoriser.transform(listD)
    sentiment = model.predict(textdata)
    # Make a list of text with sentiment.
    print(sentiment)

    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred,tokenize(text)))
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment',"tokenizedstr"])
    df = df.replace([0,1,2], ["Negative","Neutral","Positive"])

    secondop = str(text)
    second_opinion = determine_sentiment_using_blob(secondop)
    print('sentiment', df.sentiment[0])
    print('text', second_opinion)
    if second_opinion == "Neutral":
        df.sentiment[0] = "Neutral-" + df.sentiment[0]


    return df
#--------------------------------------------------------------------------------#
def determine_sentiment_using_blob(textIN):
    positive_feedbacks = []
    negative_feedbacks = []
    text =(tokenize(textIN).split())
    for feedback in text:
        feedback_polarity = TextBlob(feedback).sentiment.polarity
        if feedback_polarity>0:
            positive_feedbacks.append(feedback)
        else:
            negative_feedbacks.append(feedback)


    if len(positive_feedbacks) > len(negative_feedbacks):
        return "Positive"
    elif len(positive_feedbacks) == len(negative_feedbacks):
        return "Neutral"
    else: 
        return "Negative"
#--------------------------------------------------------------------------------#       
def tokenize(texts):

    # Remove special chars
    texts = re.sub(r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)", ' URL',texts)
    texts = emoji.demojize(texts, delimiters=("", ""))
    texts = re.sub(userPattern,' USER', texts)
    texts = texts.replace(r"’", "'")
    texts = re.sub(alphaPattern, " ", texts)
    # Remove repetitions
    texts = re.sub(sequencePattern, seqReplacePattern, texts)

    # Transform short negation form
    texts = texts.replace(r"(can't|cannot)", 'can not')
    texts = texts.replace(r"n't", ' not')

    # Remove stop words

    tokens = []
    ntokens = []
    tokens = word_tokenize(texts) 
    for i in tokens:
        if i not in stopwords:
            ntokens.append(lemma.lemmatize(i))

    return ' '.join(ntokens) 
#--------------------------------------------------------------------------------#
"""
API: Post Request on Port 500. Detect Sentiment // url:5000/sentiment/<query:string>
"""
#--------------------------------------------------------------------------------#
@app.route('/')
def my_form():
    return render_template('input.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = []
    inputT = request.form['text']
    text.append(inputT.lower())
    dfN = predict(vectoriser, LRmodel, text)
    text.clear()
    result = dfN.to_json(orient="records")
    parsed = json.loads(result)
    
    return json.dumps(parsed)

# DETECT SENTIMENT API #
class DetectSentiment(Resource):
    def get(self, text):
        texts = []
        texts.append(text.lower())
        dfN = predict(vectoriser, LRmodel, texts)
        texts.clear()
        result = dfN.to_json(orient='records')
        parsed = json.loads(result)
        print(parsed[0]["sentiment"])  
        return parsed

api.add_resource(DetectSentiment, "/sentiment/<string:text>")
#--------------------------------------------------------------------------------#
if __name__ == '__main__':
    app.run(debug=True, port = 5000, host = '0.0.0.0')

