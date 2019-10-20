# Disaster Response Pipeline Project

## Project Information
This project is part of Udacity's Data Scientist Nanodegree program. It is based on a data set by Figure Eight containing labelled disaster messages. The aim of the project is to train a classification algorithm on this dataset in order to be able to classify new text messages into one of 36 categories.

## Getting started
### Dependencies
Python 3.7+
NumPy, SciPy, Pandas, Scikit-Learn, XGBoost
NLTK
SQLalchemy
Flask, Plotly

### Running the code
1. Git clone this repository
2. Run the following commands in the project's root directory to set up the database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
3. Run the following command in the app's directory to run the web app.
    `python run.py`
4. Go to http://0.0.0.0:3001/ to view the web app and input new messages to be classified into categories.

### Author
Jörg Rechinger

### Acknowledgements
 
Udacity for providing templates and code snippets that were used to complete this project
Figure Eight for providing the messages dataset that was used to train the model

