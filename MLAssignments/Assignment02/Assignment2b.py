""" This file describes code assignment2a for COMP10200.
    Chengkang Huang, 000787205, Mohawk College, 2022
"""

# Import the libraries
from unittest import result
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def readCSV(fileName: str):
    """ Read the .csv file and combine all data into a large dataset

    Read .csv file from current folder, put all records into dataframe, and 
    return the dataset and labels.

    Arg:
        fileName(str): The string that contain the .csv file location.

    Return:
        A dataframe that contain all record from .csv file.
        The labels for all features.

    Rasies:
        File Error: File not exist or not in the correct location.
    """
    pass

    try:
        df = pd.read_csv(fileName)
        X = df.drop(['UserID'], axis=1)
        labels = (df['Art_galleries'] +
                  df['Dance_clubs'] + 
                  df['Juice_bars'] + 
                  df['Restaurants'] + 
                  df['Museums'] + 
                  df['Resorts'] + 
                  df['Picnic_spots'] + 
                  df['Beaches'] + 
                  df['Theaters'] + 
                  df['Religious_institutions']) / 10
        y = labels.apply(setLabel)
        return X, y
    except:
        print("Open file error")


def setLabel(score: float) -> str:
    """ Check the score from the data and give back the label

    Base on the rating score, compare with the enum "Rating" to set
    the label for each record.

    Arg:
        score(float): The number from each record.

    Return:
        resultStr(str): The label base on different input.

    """
    pass

    resultStr = ""
    if 0.0 <= score <= 1.0:
        resultStr = "Terrible"
    elif 1.0 < score <= 2.0:
        resultStr = "Poor"
    elif 2.0 < score <= 3.0:
        resultStr = "Average"
    elif 3.0 < score <= 4.0:
        resultStr = "VeryGood"
    elif 4.0 < score <= 5.0:
        resultStr = "Excellent"
    else:
        resultStr = "No rating scores"
    return resultStr


def dtcRun(X: list, y: list, c: str="gini", s: str="best", md: int=None, mln: int=None):
    """ Run the DecisionTreeClassifier by different parameters

    Randomly split 75% data for training, then the rest 25% data for testing,
    then put the data into DecisionTreeClassifier, finally calculate the accuracy score.

    Arg:
        X(list): The dataset.
        y(list): The labels dataset.
        c(str): The function to measure the quality of a split.
        s(str): The strategy used to choose the split at each node.
        md(int): The maximum depth of the tree.
        mln(int): Grow a tree with in best-first fashion.

    Return:
        The mean of result list.
        The original result list.

    """
    pass

    result = []
    # 5 runs
    for i in range(5):
        # Randomly split 75% data for training, then the rest 25% data for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)
        dt_clf = DecisionTreeClassifier(criterion=c, splitter=s, max_depth=md, max_leaf_nodes=mln)
        dt_clf.fit(X_train, y_train)
        y_pred = dt_clf.predict(X_test)
        result.append(accuracy_score(y_test, y_pred))
    return np.round(np.mean(np.array(result)), 6) * 100, np.round(result, 6) * 100


def main():
    """ Main code execute module """
    pass

    print("\n\n***** Look in the code for COMP10200 code assignment2b *****\n\n")
    # Code start here

    # Read data from .csv file
    fileName = "tripadvisor_review.csv"
    X, y = readCSV(fileName)

    # Test for 8 runs
    mean, runResult = dtcRun(X, y, c="gini")
    print("Criterion = gini", mean, runResult)

    mean, runResult = dtcRun(X, y, c="entropy")
    print("Criterion = entropy", mean, runResult)
    
    mean, runResult = dtcRun(X, y, s="best")
    print("Splitter = best", mean, runResult)
    
    mean, runResult = dtcRun(X, y, s="random")
    print("Splitter = random", mean, runResult)
    
    mean, runResult = dtcRun(X, y, md=5)
    print("Max_depth = 5", mean, runResult)
    
    mean, runResult = dtcRun(X, y, md=10)
    print("Max_depth = 10", mean, runResult)
    
    mean, runResult = dtcRun(X, y, mln=5)
    print("Max_leaf_nodes = 5", mean, runResult)
    
    mean, runResult = dtcRun(X, y, mln=10)
    print("Max_leaf_nodes = 10", mean, runResult)


if __name__ == '__main__':
    """ Execute main() function """
    pass

    main()
