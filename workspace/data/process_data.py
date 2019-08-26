# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """
    Loading data from csv-files.

    Parameters:
    -----------
    messages_filepath: Path to csv-file with messages_filepath
    categories_filepath: Path to csv-file with categories

    Return value:
    -------
    DateFrame with merged data.
    """

    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, how="left", on="id")

    return df

def clean_data(df):
    """
    Cleaning the data, splitting columns, setting proper columnnames.

    Parameters:
    -----------
    df: DataFrame to cleaned

    Return value:
    -------------
    Cleaned DateFrame
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.loc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[0:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(columns='categories')

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop rows with related==2
    df = df[df['related'] != 2]

    return df

def save_data(df, database_filename):
    """
    Save the data in a SQLite Database.

    Parameters:
    -----------
    df: DataFrame to save
    database_filename: Name of SQLlite Database
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')

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
