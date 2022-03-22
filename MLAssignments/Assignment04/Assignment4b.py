""" This file describes code assignment4b for COMP10200.
    Chengkang Huang, 000787205, Mohawk College, 2022
"""

# Import the libraries
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
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

    print("\n\n***** Look in the code for COMP10200 code assignment4b *****\n")
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
            clf_model = DecisionTreeClassifier()
            clf_model.fit(X_train, y_train)
            clf_y_pred = clf_model.predict(X_test)
            clf_accuracy_score = np.round(
                accuracy_score(y_test, clf_y_pred), 2) * 100
            mlp_clf = MLPClassifier(verbose=False)
            mlp_clf.fit(X_train, y_train)
            mlp_y_pred = mlp_clf.predict(X_test)
            mlp_accuracy_score = np.round(
                accuracy_score(y_test, mlp_y_pred), 2) * 100
            print('File: {}_{}.csv\nDecision Tree: {}% Accuracy\nMLP: hidden layers = {}, LR = {}, tol = {}\n{} Accuracy, {} iterations\n'.format(
                student_id, i,
                clf_accuracy_score,
                mlp_clf.hidden_layer_sizes,
                mlp_clf.learning_rate,
                mlp_clf.tol,
                mlp_accuracy_score,
                mlp_clf.max_iter))

    except:
        print("Program run failed")


if __name__ == '__main__':
    """ Execute main() function """
    pass

    main()
