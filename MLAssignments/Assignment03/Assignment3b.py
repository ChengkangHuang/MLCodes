""" This file describes code assignment3b for COMP10200.
    Chengkang Huang, 000787205, Mohawk College, 2022
"""

"""
    (Dataset Link: "https://www.kaggle.com/priyaduttbhatt/text-classifier?select=train_data.csv")
"""

# Import the libraries
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline


def readCSV(fileName: str):
    """ Read the .csv file and combine all data into a large dataset

    Read .csv file from current folder, put all records into dataframe, and 
    return the dataset and labels.

    Arg:
        fileName(str): The string that contain the .csv file location.

    Return:
        A dataset that contain record from .csv file.
        The labels for features.

    Rasies:
        File Error: File not exist or not in the correct location.
    """
    pass

    try:
        df = pd.read_csv(fileName)
        dataset = df['transcription']
        labels = df['action']
        return dataset, labels
    except:
        print("Open file error")


def get_runs_result(X: list, y: list, s_w: str = None, a: str = 'word', n_r: tuple = (1, 1)):
    """ Run the CountVectorizer and MultinomialNB and get the result

    Randomly split 75% data for training, then the rest 25% data for testing.

    Arg:
        X(list): The dataset.
        y(list): The labels dataset.

    Return:
        v_mnb_model_score(float): the score of v_mnb_model.
    """
    pass

    c_vectorizer = CountVectorizer(stop_words=s_w, analyzer=a, ngram_range=n_r)
    dt_matrix = c_vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    mnb_model = MultinomialNB()
    v_mnb_model = Pipeline(
        steps=[('vectorizer', c_vectorizer), ('classifier', mnb_model)])
    v_mnb_model.fit(X_train, y_train)
    v_mnb_probs = v_mnb_model.predict_proba(X_test).max(axis=1)
    low_confidence = v_mnb_probs < 0.9
    y_true = low_confidence != True
    y_pred = v_mnb_model.predict(X_test)
    v_mnb_model_score = round(v_mnb_model.score(X_test, y_test), 4)
    v_mnb_model_prec_score = precision_score(y_true, y_pred, average='micro')
    v_mnb_model_recall_score = recall_score(y_true, y_pred, average='micro')
    return v_mnb_model_score, v_mnb_model_prec_score, v_mnb_model_recall_score


def main():
    """ Main code execute module """
    pass

    print("\n\n***** Look in the code for COMP10200 code assignment3b *****\n\n")
    # Code start here

    # Read data from .csv file
    dirPath = os.path.abspath('.')
    fileName = dirPath + "\\MLAssignments\\Assignment03\\train_data.csv"
    X, labels = readCSV(fileName)
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # Mutiple runs with different parameters
    score, prec_score, recall_score = get_runs_result(X, y)
    print("The micro-averaged accuracy is {}, precision is {}, and recall is {}"
          .format(np.round(score, 4), np.round(prec_score, 6), np.round(recall_score, 6)))

    score, prec_score, recall_score = get_runs_result(X, y, s_w='english', a='word')
    print("The micro-averaged accuracy is {}, precision is {}, and recall is {}"
          .format(np.round(score, 4), np.round(prec_score, 6), np.round(recall_score, 6)))
    
    score, prec_score, recall_score = get_runs_result(X, y, s_w='english', a='char')
    print("The micro-averaged accuracy is {}, precision is {}, and recall is {}"
          .format(np.round(score, 4), np.round(prec_score, 6), np.round(recall_score, 6)))
    
    score, prec_score, recall_score = get_runs_result(X, y, s_w='english', a='char_wb')
    print("The micro-averaged accuracy is {}, precision is {}, and recall is {}"
          .format(np.round(score, 4), np.round(prec_score, 6), np.round(recall_score, 6)))

    score, prec_score, recall_score = get_runs_result(X, y, s_w='english', a='word', n_r=(2, 2))
    print("The micro-averaged accuracy is {}, precision is {}, and recall is {}"
          .format(np.round(score, 4), np.round(prec_score, 6), np.round(recall_score, 6)))

    score, prec_score, recall_score = get_runs_result(X, y, s_w='english', a='char_wb', n_r=(2, 2))
    print("The micro-averaged accuracy is {}, precision is {}, and recall is {}"
          .format(np.round(score, 4), np.round(prec_score, 6), np.round(recall_score, 6)))

    score, prec_score, recall_score = get_runs_result(X, y, s_w='english', a='char', n_r=(2, 2))
    print("The micro-averaged accuracy is {}, precision is {}, and recall is {}"
          .format(np.round(score, 4), np.round(prec_score, 6), np.round(recall_score, 6)))

    score, prec_score, recall_score = get_runs_result(X, y, a='word', n_r=(2, 2))
    print("The micro-averaged accuracy is {}, precision is {}, and recall is {}"
          .format(np.round(score, 4), np.round(prec_score, 6), np.round(recall_score, 6)))


if __name__ == '__main__':
    """ Execute main() function """
    pass

    main()
