import sys
# import libraries
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

class SortVectorizer():
    """
    Auxiliar estimator to sort the indices of the TfidTransformer
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.sort_indices()
        return X

def load_data(database_filepath):
    """
    Loads the data from the specified database
    :param database_filepath (str): Database filepath
    :return: X, Y, columns
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM message_category", con=engine.connect())
    X = df.message.values
    Y = df[list(set(df.columns) - {'message','original', 'genre', 'id'})]
    return X, Y, Y.columns

def tokenize(text):
    """
    This function uses nltk to case normalize, lemmatize, and tokenize the text
    :param text (str):
    :return: Array of clean tokens
    """
    text = re.sub('[\W\d_-]', ' ', text)
    text = re.sub('^\w{1,2} | \w{1,2} | \w{1,2}$', ' ', text)
    tokens = word_tokenize(text) 
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    This functions returns a model using pipeline and GridSearchCV
    :return:
    """
    clf = AdaBoostClassifier()
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('sortv', SortVectorizer()),
        ('clf',  MultiOutputClassifier(clf, n_jobs=-1))
    ])
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__learning_rate': [1, 2]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test):
    """
    This function evaluates the model using classification_report for each category
    :param model:
    :param X_test:
    :param Y_test:
    :return:
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Saves the model using joblib.dump instead of pickle library
    :param model(sklearn model): Trained model
    :param model_filepath(str): File path for the classifier
    :return: None
    """
    joblib.dump(model, model_filepath)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()