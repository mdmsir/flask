import pandas as pd
from flask import *
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
import re
from nltk.corpus import stopwords
from distutils.log import debug
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from fileinput import filename
from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = os.path.join('static', 'uploads')
 
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}
 
app = Flask(__name__)
 
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
app.secret_key = 'This is your secret key to utilize session in Flask'
 
 
@app.route('/', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
      # upload file flask
        f = request.files.get('file')
 
        # Extracting uploaded file name
        data_filename = secure_filename(f.filename)
 
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],data_filename))
 
        session['uploaded_data_file_path'] =os.path.join(app.config['UPLOAD_FOLDER'],data_filename)
 
        return render_template('index2.html')
    return render_template("index.html")

@app.route('/show_data')
def showData():
    # Load your dataset
    # Assuming you have a CSV file with columns 'text' and 'label'
    data_file_path = session.get('uploaded_data_file_path', None)
    data = pd.read_csv(data_file_path,encoding='unicode_escape')
    #data = pd.read_csv(data_filename)

#
    # Data Preprocessing
    # Assuming 'text' column contains the text data and 'label' column contains sentiment labels (positive/negative)
#   X = data['Summary']
#   y = data['label']

    # Simple Tokenization and Lowercasing for Preprocessing
#   X = X.str.lower().str.split()

    
    # Convert the list of tokens back to a string for TfidfVectorizer
#    X = X.apply(lambda x: ' '.join(x))

    # Split the data into training and testing sets
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Extraction using TF-IDF
#    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
#    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
#    X_test_tfidf = tfidf_vectorizer.transform(X_test)

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

    
    # Feature Selection using SGD (Stochastic Gradient Descent) with L1 regularization
    sgd = SGDClassifier(loss='hinge', penalty='l1', alpha=1e-3, random_state=42)
    sgd.fit(X_train_tfidf, y_train)
    feature_selector = SelectFromModel(sgd, prefit=True)
    X_train_selected = feature_selector.transform(X_train_tfidf)
    X_test_selected = feature_selector.transform(X_test_tfidf)

    # SVM Model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_selected, y_train)
    svm_predictions = svm_model.predict(X_test_selected)
    svm_accuracy = accuracy_score(y_test, svm_predictions)

    # Decision Tree Model (ID3)
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train_selected, y_train)
    dt_predictions = dt_model.predict(X_test_selected)
    dt_accuracy = accuracy_score(y_test, dt_predictions)

# AdaBoost Model with Decision Tree as base estimator
    adaboost_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
    adaboost_model.fit(X_train_selected, y_train)
    adaboost_predictions = adaboost_model.predict(X_test_selected)
    adaboost_accuracy = accuracy_score(y_test, adaboost_predictions)

# Additional Evaluation Metrics
    #print("\nSVM Classification Report:")
    #print(classification_report(y_test, svm_predictions))
    cv=classification_report(y_test, svm_predictions,output_dict=True)
    cd=classification_report(y_test, dt_predictions,output_dict=True)
    ca=classification_report(y_test, adaboost_predictions,output_dict=True)

    
    # Convert report to DataFrame
    df = pd.DataFrame(cv).transpose()
    df1 = pd.DataFrame(cd).transpose()
    df2 = pd.DataFrame(ca).transpose()

    # Convert DataFrame to HTML table
    html_table = df.to_html()
    html_table1 = df1.to_html()
    html_table2 = df2.to_html()
    
    return render_template('index.html', text1='svm_accuracy',text2=svm_accuracy,text3='dt_accuracy', text4=dt_accuracy,text5='adaboost_accuracy',text6=adaboost_accuracy,table=html_table,table1=html_table1,table2=html_table2)



if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5003, threaded=True)
