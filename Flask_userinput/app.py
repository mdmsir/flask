import pandas as pd

from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
import re
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report



set(stopwords.words('english'))

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    stop_words = stopwords.words('english')
    
    #convert to lowercase
    text1=request.form['text1']
    val=[text1]

    # Load your dataset
    # Assuming you have a CSV file with columns 'text' and 'label'
    data = pd.read_csv('twitter_train.csv',encoding='unicode_escape')

    # Data Preprocessing
    # Assuming 'text' column contains the text data and 'label' column contains sentiment labels (positive/negative)
    # X = data['Summary']
    # y = data['label']

    # Simple Tokenization and Lowercasing for Preprocessing
    # X = X.str.lower().str.split()

    # Convert the list of tokens back to a string for TfidfVectorizer
    # X = X.apply(lambda x: ' '.join(x))

    
    X = data['Summary'].str.lower().str.split()  # Tokenization and lowercasing
    X = X.apply(lambda x: ' '.join(x) if isinstance(x, list) else x)  # Convert tokens to string
    y = data['label']

    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handling NaN values

    X_train.fillna('', inplace=True)
    X_train.dropna(inplace=True)

    X_test.fillna('', inplace=True)
    X_test.dropna(inplace=True)

    
    # Feature Extraction using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    X_val_tfidf = tfidf_vectorizer.transform(val)

    # Feature Selection using SGD (Stochastic Gradient Descent) with L1 regularization
    sgd = SGDClassifier(loss='hinge', penalty='l1', alpha=1e-3, random_state=42)
    sgd.fit(X_train_tfidf, y_train)
    feature_selector = SelectFromModel(sgd, prefit=True)
    X_train_selected = feature_selector.transform(X_train_tfidf)
    X_test_selected = feature_selector.transform(X_test_tfidf)
    X_val_selected = feature_selector.transform(X_val_tfidf)

    # SVM Model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_selected, y_train)
    svm_predictions = svm_model.predict(X_test_selected)
    svm_predictions_val = svm_model.predict(X_val_selected)

    # DT Model
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train_selected, y_train)
    dt_predictions = dt_model.predict(X_test_selected)
    dt_predictions_val = dt_model.predict(X_val_selected)

    # AdaBoost Model with Decision Tree as base estimator
    adaboost_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
    adaboost_model.fit(X_train_selected, y_train)
    adaboost_predictions = adaboost_model.predict(X_test_selected)
    adaboost_predictions_val = adaboost_model.predict(X_val_selected)
    

    return render_template('form.html', text1=val,text2=svm_predictions_val,text3=dt_predictions_val,text4=adaboost_predictions_val)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
