""" This file describes code assignment4a for COMP10200.
    Chengkang Huang, 000787205, Mohawk College, 2022
"""

"""
    The output:
    ***** Look in the code for COMP10200 code assignment4a *****
    
    Result:
    000787205_1.csv: 98.0% W: [-24.51966669 -42.12227391 -12.18679875 -41.95810518 -19.06206528
    -26.29115405  -8.63336545 -23.73941121 -27.48822523] T: 21.0
    000787205_2.csv: 99.0% W: [-36.05178883 -19.27771373 -38.40779154 -47.56156856 -27.55565417
    -4.19771316 -18.87844415] T: -30.0
    000787205_3.csv: 52.0% W: [ -68.01069584  -43.71001547   42.79329981 -243.63052738   24.646952
    225.97465641 -475.27546073  141.23257852] T: 79.0
    000787205_4.csv: 49.0% W: [ 259.49705408  345.21088067  -68.60661176   52.1707237  -129.96107081
    -27.17487031   57.24321269] T: 41.0

    ---------------------------------------------------------------------------------------------
    
    Report

        My student id is 000787205, base on the student id, there are four csv files will being 
    test; relating the testing result in this assignment:
        000787205_1.csv is linearly separable datasets
        000787205_2.csv is linearly separable datasets
        000787205_3.csv is not linearly separable datasets
        000787205_4.csv is not linearly separable datasets

"""

# Import the libraries
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


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
        datasets = data[:, :len(data[0]) - 1]
        labels = data[:, len(data[0]) - 1:]
        return datasets, np.ravel(labels)
    except:
        print("Open file error")


def main():
    """ Main code execute module """
    pass

    print("\n\n***** Look in the code for COMP10200 code assignment4a *****\n")
    # Code start here

    try:
        # Set the absolute path
        dirPath = os.path.abspath(".")
        student_id = 787205

        """ Run the program 4 times, because there are 4 files that relative with
        student id (000787205) """

        print("Result:")
        for i in range(1, 5):
            # Please replace the path if needed
            fileName = dirPath + "\\MLAssignments\\Assignment04\\" + \
                str(student_id) + "_" + str(i) + ".csv"
            X, y = readCSV(fileName)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=32)
            p_clf = Perceptron(verbose=False)
            p_clf.fit(X_train, y_train)
            y_pred = p_clf.predict(X_test)
            print("000{}_{}.csv: {}% W: {} T: {}".format(
                student_id, i, 
                np.round(accuracy_score(y_test, y_pred), 2) * 100, 
                p_clf.coef_[0], 
                p_clf.intercept_[0]))

    except:
        print("Program run failed")


if __name__ == '__main__':
    """ Execute main() function """
    pass

    main()
