import json
import os
from flask import Flask,jsonify,request
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import pickle
from sklearn.feature_extraction.text import CountVectorizer
#from su_predictor import suicide_predictor

app = Flask(__name__)
CORS(app)
@app.route("/price/",methods=['GET'])
def return_price():
    text_list = []
    #t = request.args.get('text')
    text_list = request.args.getlist('text')
    #text_list.append(t)
    
    text_list = [i.lower() for i in text_list]

    # Tokenization
    text_list = [word_tokenize(i) for i in text_list]
    
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    for index, entry in enumerate(text_list):
        final_words = []
        word_lemmatized = WordNetLemmatizer()
    
        for word, tag in pos_tag(entry):
            if word not in stopwords.words('english') and word.isalpha():
                word_final = word_lemmatized.lemmatize(word, tag_map[tag[0]])
                final_words.append(word_final)
        text_list[index] = str(final_words)      
        
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
    tfidf = transformer.fit_transform(loaded_vec.transform(text_list))
        
    with open('ml_svm.pkl', 'rb') as handle:
        model = pickle.load(handle)

    preds = model.predict(tfidf)
    
    preds = preds.tolist()
    json.dumps({"prediction": preds})    
    
    pred_dict = {
                'model':'svm',
                'price': preds,
                }
    return jsonify(pred_dict)

@app.route("/",methods=['GET'])
def default():
    return "<h1> Welcome to suicide predictor <h1>"

if __name__ == "__main__":
    app.run() 