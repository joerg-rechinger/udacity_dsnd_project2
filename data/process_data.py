# import libraries
import sys
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    cwd = os.getcwd()
    messages=pd.read_csv(os.path.join(cwd,messages_filepath))
    categories=pd.read_csv(os.path.join(cwd,categories_filepath))
    df = messages.merge(categories, on='id', how='outer')
    return df

def clean_data(df):
    """
    create a dataframe of the 36 individual category columns
    select the first row of the categories dataframe
    use this row to extract a list of new column names for categories.
    rename the columns of `categories`
    set each value to be the last character of the string
    convert column from string to numeric
    concatenate the original dataframe with the new `categories` dataframe
    drop duplicates
    """
    categories_cols = df['categories'].str.split(";",expand=True)
    row = categories_cols.iloc[0].to_list()
    category_colnames = [i[0:-2] for i in row]
    categories_cols.columns = category_colnames
    for column in categories_cols:
        categories_cols[column] = categories_cols[column].astype(str).str[-1:]
        categories_cols[column] = pd.to_numeric(categories_cols[column])
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories_cols], axis=1)
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename, table_name):
    engine = create_engine(str('sqlite:///')+str(database_filename), encoding='UTF-8')
    df.to_sql(table_name, engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, 'disaster_messages')

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
