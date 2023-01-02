""" This file describes code assignment5b for COMP10200.
    Chengkang Huang, 000787205, Mohawk College, 2022
"""

# Import the libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def readCSV(fileName: str):
    """ Read the .csv file and combine all data into a large dataset

    Read .csv file from current folder, put all records into numpy array, and 
    return the dataset and labels.

    Arg:
        fileName(str): The string that contain the .csv file location.

    Return:
        A numpy array that contain all record from .csv file.
        The labels for all features.

    Rasies:
        File Error: File not exist or not in the correct location.
    """
    pass

    try:
        data = np.loadtxt(fileName, delimiter=',', skiprows=1)
        datasets = data[:, 1:]
        labels = data[:, 0:1]
        return datasets, np.ravel(labels)
    except:
        print("Open file error")


def main():
    """ Main code execute module """
    pass

    print("\n\n***** Look in the code for COMP10200 code assignment5b *****\n")
    # Code start here

    try:
        # Set the absolute path
        dirPath = os.path.abspath(".")

        print("Result:")
        # Please replace the path if needed
        fileName = dirPath + "\\MLAssignments\\Assignment05\\" + "accelerometer.csv"
        X, y = readCSV(fileName)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=32)
        lin_reg = LinearRegression().fit(X_train, y_train)
        y_pred = lin_reg.predict(X_test)
        print("The training dataset size is: ", len(X_train))
        print("The testing dataset size is: ", len(X_test))
        print("The feature size is: ", len(X_train[0]))
        print("The accuracy score is: ", accuracy_score(y_test, y_pred))
        print("The RSS is: ", np.sum(np.square(y_pred - y_test)))
        print("The correlation coefficient is: ", np.corrcoef(y_pred, y_test)[0, 1])
        print("The weighted rss is: ", np.sum(np.square(y_pred - y_test)) / len(y_test))
        print("The intercept is: ", lin_reg.intercept_)

    except:
        print("Program run failed")


if __name__ == '__main__':
    """ Execute main() function """
    pass

    main()
