# import libraries
import sys

# Storage
import pandas as pd
from sqlalchemy import create_engine

# NLP
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.externals import joblib

def load_data(database_filepath):
    """
    Load data from SQLlite Database. Also remove columns with just one value
    because this may crash the classifier.

    Parameters:
    -----------
    database_filepath: path to SQLlite Database

    Return value:
    -------------
    X: DataFrame with features
    Y: DataFrame with targets
    category_names: Names of the targets
    """

    def drop_single_value_columns(data):
        """
        Drop any column which doesn't contain two diffent values

        Parameters:
        -----------
        data: DataFrame

        Return value:
        -------------
        DataFrame with column(s) removed
        """

        data.columns[data.nunique(axis=0) != 2]
        data.drop(columns=data.columns[data.nunique(axis=0) != 2], inplace=True)
        return data


    engine = create_engine('sqlite:///' + database_filepath)
    connection = engine.connect()
    df = pd.read_sql_table("messages", con=connection)
    X = df.iloc[:, 1]
    Y = drop_single_value_columns(df.iloc[:, 4:])
    category_names = list(Y.columns)

    return X, Y, category_names

# That's a work around to because when defining stop_words inside of tokenize()
# I can only use n_jobs=1 in GridSearchCV. Otherwise I get the following error:
# _pickle.PicklingError: Could not pickle the task to send it to the workers.
stop_words = stopwords.words('english')

def tokenize(text):
    """
    Tokenize text using word_tokenize and WordNetLemmatizer

    Parameters:
    -----------
    text: String with feature_extraction

    Return value:
    -------------
    List of tokenized text
    """

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize andremove stop words
    # stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Building a pipeline to transform input data and classify with SVC

    Return value:
    -------------
    Model for Cross-validation
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('mcfl', MultiOutputClassifier(estimator=SVC()))
    ])

    parameters = {
        # 'mcfl__estimator__kernel' : ['linear', 'rbf'],
        'mcfl__estimator__kernel' : ['linear'],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10, cv=2, \
        n_jobs=-1, scoring='f1_micro')

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred))

def save_model(model, model_filepath):
    """
    Save best model to disk

    Parameters:
    -----------
    model: GridSearchCV
    model_filepath: Path for storage
    """

    joblib.dump(model.best_estimator_, model_filepath, compress = 1)


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
