""" This file describes code assignment3a for COMP10200.
    Chengkang Huang, 000787205, Mohawk College, 2022
"""

"""
    The output:
    ***** Look in the code for COMP10200 code assignment3a *****


    50 runtimes collection result:

    kNN average score:  0.9817
    kNN correct probability score:  0.9669
    kNN incorrect probability score:  0.0331
    CLF average score:  0.9736
    CLF correct probability score:  0.9795
    CLF incorrect probability score:  0.0205
    NB average score:  0.9557
    NB correct probability score:  0.9198
    NB incorrect probability score:  0.0802
    
    ---------------------------------------------------------------------------------------------
    
    The report
        In this program, "tripadvisor_review.csv" is the dataset that being used, 
    get_muti_runs_result() run KNeighborsClassifier, DecisionTreeClassifier and GaussianNB to get 
    the results. Base on the result of my test cases, after 50 runtimes, the average accuracy of 
    KNeighborsClassifier is around 0.9817, and the average accuracy of DecisionTreeClassifier is 
    around 0.9736, the GaussianNB's average accuracy is around 0.9557, which is a lower than 
    KNeighborsClassifier and DecisionTreeClassifier. Comparing the probability score of correct of 
    KNeighborsClassifier, DecisionTreeClassifier and GaussianNB, DecisionTreeClassifier have the 
    highest score 0.9795, then 0.9669 for KNeighborsClassifier, and the lowest 0.9198 for GaussianNB. 
    After that, the probability score of incorrect of KNeighborsClassifier, DecisionTreeClassifier 
    and GaussianNB respectively are 0.0331, 0.0205 and 0.0802.
        Overall, KNeighborsClassifier have the best proformence during 50 times run, but the 
    correct probability is a bit lower than DecisionTreeClassifier, also, GaussianNB have the 
    worst proformence in correct and incorrect probability. The possible reason of GaussianNB have
    the worst result is because the assumption of independence of sample attributes is used, the 
    effect is not good if the sample attributes are correlated. So that the GaussianNB is more 
    used in text classification, fraud detection, email filtering.

    (Dataset Link: "https://archive.ics.uci.edu/ml/datasets/Travel+Reviews")
"""

# Import the libraries
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB


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
        label = (df['Art_galleries'] +
                 df['Dance_clubs'] +
                 df['Juice_bars'] +
                 df['Restaurants'] +
                 df['Museums'] +
                 df['Resorts'] +
                 df['Picnic_spots'] +
                 df['Beaches'] +
                 df['Theaters'] +
                 df['Religious_institutions']) / 10
        labels = label.apply(setLabel)
        return X, labels
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


def get_muti_runs_result(X: list, y: list):
    """ Run the KNeighborsClassifier, DecisionTreeClassifier and 
        GaussianNB 50 times to get the result

    Randomly split 75% data for training, then the rest 25% data for testing,
    then put the data into KNeighborsClassifier, DecisionTreeClassifier and GaussianNB, 
    finally calculate the result.

    Arg:
        X(list): The dataset.
        y(list): The labels dataset.

    Return:
        The list of 
        knn_avg_score, 
        knn_prob_score_correct, 
        knn_prob_score_incorrect,
        clf_avg_score, 
        clf_prob_score_correct, 
        clf_prob_score_incorrect, 
        nb_avg_score, 
        nb_prob_score_correct, 
        nb_prob_score_incorrect.

    """
    pass

    # Array to store each runtime result
    knn_avg_score_result = []
    knn_prob_score_correct_result = []
    knn_prob_score_incorrect_result = []
    clf_avg_score_result = []
    clf_prob_score_correct_result = []
    clf_prob_score_incorrect_result = []
    nb_avg_score_result = []
    nb_prob_score_correct_result = []
    nb_prob_score_incorrect_result = []
    
    # 50 runs
    for i in range(50):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=None)

        # Apply KNeighborsClassifier
        knn_model = KNeighborsClassifier(3)
        knn_model.fit(X_train, y_train)
        knn_score = knn_model.score(X_test, y_test)
        knn_probs = knn_model.predict_proba(X_test).max(axis = 1)
        low_confidence_knn = knn_probs < 0.9
        knn_avg_score_result.append(knn_score)
        knn_prob_score_correct_result.append((low_confidence_knn != True).sum() / X_test.shape[0])
        knn_prob_score_incorrect_result.append((low_confidence_knn == True).sum() / X_test.shape[0])

        # Apply DecisionTreeClassifier
        clf_model = DecisionTreeClassifier(max_depth=3)
        clf_model.fit(X_train, y_train)
        clf_score = clf_model.score(X_test, y_test)
        clf_probs = clf_model.predict_proba(X_test).max(axis = 1)
        low_confidence_clf = clf_probs < 0.9
        clf_avg_score_result.append(clf_score)
        clf_prob_score_correct_result.append((low_confidence_clf != True).sum() / X_test.shape[0])
        clf_prob_score_incorrect_result.append((low_confidence_clf == True).sum() / X_test.shape[0])

        # Apply GaussianNB
        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)
        nb_score = nb_model.score(X_test, y_test)
        nb_probs = nb_model.predict_proba(X_test).max(axis = 1)
        low_confidence_nb = nb_probs < 0.9
        nb_avg_score_result.append(nb_score)
        nb_prob_score_correct_result.append((low_confidence_nb != True).sum() / X_test.shape[0])
        nb_prob_score_incorrect_result.append((low_confidence_nb == True).sum() / X_test.shape[0])

    return np.round(np.mean(np.array(knn_avg_score_result)), 4),\
           np.round(np.mean(np.array(knn_prob_score_correct_result)), 4),\
           np.round(np.mean(np.array(knn_prob_score_incorrect_result)), 4),\
           np.round(np.mean(np.array(clf_avg_score_result)), 4),\
           np.round(np.mean(np.array(clf_prob_score_correct_result)), 4),\
           np.round(np.mean(np.array(clf_prob_score_incorrect_result)), 4),\
           np.round(np.mean(np.array(nb_avg_score_result)), 4),\
           np.round(np.mean(np.array(nb_prob_score_correct_result)), 4),\
           np.round(np.mean(np.array(nb_prob_score_incorrect_result)), 4),\


def main():
    """ Main code execute module """
    pass

    print("\n\n***** Look in the code for COMP10200 code assignment3a *****\n\n")
    # Code start here

    try:
        # Read data from .csv file
        dirPath = os.path.abspath(".")
        fileName = dirPath + "\\MLAssignments\\Assignment03\\tripadvisor_review.csv"
        X, labels = readCSV(fileName)
        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)

        knn_avg_score, knn_prob_score_correct, knn_prob_score_incorrect,\
        clf_avg_score, clf_prob_score_correct, clf_prob_score_incorrect,\
        nb_avg_score, nb_prob_score_correct, nb_prob_score_incorrect = get_muti_runs_result(X, y)

        print("50 runtimes collection result:\n")
        print("kNN average score: ", knn_avg_score)
        print("kNN correct probability score: ", knn_prob_score_correct)
        print("kNN incorrect probability score: ", knn_prob_score_incorrect)
        print("CLF average score: ", clf_avg_score)
        print("CLF correct probability score: ", clf_prob_score_correct)
        print("CLF incorrect probability score: ", clf_prob_score_incorrect)
        print("NB average score: ", nb_avg_score)
        print("NB correct probability score: ", nb_prob_score_correct)
        print("NB incorrect probability score: ", nb_prob_score_incorrect)
    except:
        print("Program run failed")


if __name__ == '__main__':
    """ Execute main() function """
    pass

    main()
