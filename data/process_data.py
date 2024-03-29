import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This function loads loads the data from two CSV files and returns it merged in ona DataFrame
    :param messages_filepath (str): Message CSV filepath
    :param categories_filepath (str): Categories CSV filepath
    :return:  pandas.DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='inner')
    return df

def clean_data(df):
    """
    Transform the categories field into multiple columns and then drops the duplicates rows
    :param df (pandas.DataFrame): Dataframe with messages an their categories related
    :return: pandas.DataFrame
    """
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda a: a[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = [val[-1] for val in categories[column]]
        categories[column] = categories[column].astype('int')
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    return df
    
def save_data(df, database_filename):
    """
    Saves the DataFrame into a sqlite database
    :param df(pandas.DataFrame): Dataframe to store
    :param database_filename: Name of the database
    :return: None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('message_category', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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