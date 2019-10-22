# import libraries
import sys
import pandas as pd
import numpy as np
import os
import re
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import nltk
nltk.download(['punkt', 'wordnet'])
import pickle


def load_data(database_filepath):
    """
    Loads and prepares data from db table
    Several '2' values in first column of df > needs to be cleaned (Input found here: https://knowledge.udacity.com/questions/56317)

    Args:
        database_filepath: filepath to the db where disaster_messages table is stored

    Returns:
        X: independent variable that is used as input for classifier
        Y: dependent variables that we are trying to classify
        categories: category names of the 36 different categories

    """
    engine = create_engine(str('sqlite:///')+str(database_filepath))
    df = pd.read_sql_table('disaster_messages', engine)
    X = df.message.values
    Y = df.iloc[:,4:]
    categories = list(Y.columns)
    Y=Y.values

    for i, col in enumerate(Y):
        for j, row in enumerate(col):
            if col[j] not in [0, 1]:
                Y[i][j] = 1
    return X, Y, categories

def tokenize(text):
    """
    Tokenizes and lemmatizes the text messages

    Args:
        text: text message to tokenize

    Returns:
        list of tokenized and lemmatized words from the text message
    """
    text=re.sub(r"[^A-Za-z0-9\-]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    """
    Builds the pipeline and performs grid search on a set of parameters
    Input for optimizing xgboost found here: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

    Args:
        None

    Returns:
        GridSearch Object that can then be trained on training data
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(XGBClassifier(learning_rate =0.1,
         n_estimators=140,
         max_depth=5,
         min_child_weight=1,
         gamma=0,
         subsample=0.8,
         colsample_bytree=0.8,
         objective= 'binary:logistic',
         nthread=4,
         scale_pos_weight=1,
         seed=27)))
    ])

    parameters = {
        'clf__estimator__min_child_weight': range(3,6,2),
        'clf__estimator__max_depth': range(3,8,2)
    }

    cv = GridSearchCV(pipeline, cv=5, param_grid=parameters, verbose=1, n_jobs=4)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Predicts based on the trained model
    Args:
        model: the trained model
        X_test: X-values of the test data set
        Y_test: Y-values of the test data set
        category_names: names of the 36 categories that messages can be classified into
    Returns:
        None
    """
    y_pred=model.predict(X_test)
    for i in range(len(category_names)):
        print(classification_report(Y_test[:, i], y_pred[:, i]))



def save_model(model, model_filepath):
    """
    Saves model as pickle file to use for web app
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)




def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        Y=Y.astype('int')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)
        print(model.best_score_, model.best_params_)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
