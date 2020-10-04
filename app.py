import pandas as pd
import pickle
from flask import Flask, request, render_template
from flask_restful import Api, Resource
from textblob import TextBlob
import re
import emoji
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()

stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", 
"yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", 
"their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
"was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", 
"and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", 
"between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", 
"on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", 
"all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", 
"same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
stopwords.remove('not')
stopwords.remove('nor')
stopwords.remove('no')
# Do this first, that'll do something eval() 
# to "materialize" the LazyCorpusLoader
next(swn.all_senti_synsets()) 


pattern = '@\S+|https?:\S+|http?:\S|[^A-Za-z]+|com|net'
urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
userPattern       = '@[^\s]+'
alphaPattern      = "[^a-zA-Z0-9]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

import json
file = open('LR.pkl', 'rb')
LRmodel = pickle.load(file)
file.close()

file = open('vectoriser.pkl','rb')
vectoriser = pickle.load(file)
file.close()

app = Flask(__name__, template_folder='template')
api = Api(app)

def predict(vectoriser, model, text):
    # Predict the sentiment
    listD = tokenize(str(text).lower())
    listD = [listD]
    textdata = vectoriser.transform(listD)
    sentiment = model.predict(textdata)
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
        
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1,2], ["Negative","Neutral","Positive"])
    print(df.sentiment)
    print(determine_sentiment_using_blob(text))
    return df

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
    tokens = list(texts) 
    for i in tokens:
        if i not in stopwords:
            ntokens.append(lemma.lemmatize(i))

    return ' '.join(ntokens)  




if __name__ == '__main__':
    app.run(debug=True, port = 5000)

