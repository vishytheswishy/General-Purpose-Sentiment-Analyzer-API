from tokenization import tokenize
import pandas as pd
import pickle
from flask import Flask, request, render_template
from flask_restful import Api, Resource
from textblob import TextBlob

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

if __name__ == '__main__':
    app.run(debug=True, port = 5000)

