""" This file describes code assignment2a for COMP10200.
    Chengkang Huang, 000787205, Mohawk College, 2022
"""

# Import the libraries
import csv
import numpy as np
from matplotlib import pyplot as plt


def readCSV(fileName: str):
    """ Read the .csv file and combine all data into a large dataset

    Read .csv file from current folder, append each record data into each
    feature name array, finally combine them into a new large numpy array.

    Arg:
        fileName(str): The string that contain the .csv file location.

    Return:
        A new large numpy array that contain all record from .csv file.
        Ex:
        np.array([UserId, Art_galleries, Dance_clubs, Juice_bars, Restaurants, Museums, Resorts, Picnic_spots, Beaches, Theaters, Religious_institutions, labels])

    Rasies:
        File Error: File not exist or not in the correct location.
        Read Row Error: Something wrong with the record data in the .csv file.
    """
    pass

    UserId = []
    Art_galleries = []
    Dance_clubs = []
    Juice_bars = []
    Restaurants = []
    Museums = []
    Resorts = []
    Picnic_spots = []
    Beaches = []
    Theaters = []
    Religious_institutions = []
    labels = []

    try:
        with open(fileName) as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    UserId.append(int(row['UserID']))
                    Art_galleries.append(float(row['Art_galleries']))
                    Dance_clubs.append(float(row['Dance_clubs']))
                    Juice_bars.append(float(row['Juice_bars']))
                    Restaurants.append(float(row['Restaurants']))
                    Museums.append(float(row['Museums']))
                    Resorts.append(float(row['Resorts']))
                    Picnic_spots.append(float(row['Picnic_spots']))
                    Beaches.append(float(row['Beaches']))
                    Theaters.append(float(row['Theaters']))
                    Religious_institutions.append(
                        float(row['Religious_institutions']))
                    labels.append(str(evaluation((float(row['Art_galleries']) +
                                                  float(row['Dance_clubs']) +
                                                  float(row['Juice_bars']) +
                                                  float(row['Restaurants']) +
                                                  float(row['Museums']) +
                                                  float(row['Resorts']) +
                                                  float(row['Picnic_spots']) +
                                                  float(row['Beaches']) +
                                                  float(row['Theaters']) +
                                                  float(row['Religious_institutions'])) / 10)))
                except:
                    print("Read row error")
            print("Successfully read the data")
    except:
        print("Open file error")

    return np.array([labels]), np.array([Art_galleries, Dance_clubs, Juice_bars, Restaurants, Museums, Resorts, Picnic_spots, Beaches, Theaters, Religious_institutions])


def evaluation(score: float) -> str:
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


def kNNClassifier(trainingData, trainingLabels, testingData, kValue):
    topData = np.array([])
    for i in range(len(testingData[0])):
        distanceArray = np.sqrt(((trainingData - testingData[:, i:i + 1]) ** 2).sum(axis=0))
        min_dist_point = trainingLabels[:, distanceArray.argmin()]
        topData = trainingData[:, distanceArray.argsort()[:kValue]]
    return topData


def main():
    """ Main code execute module """
    pass

    print("\n\n***** Look in the code for COMP10200 code assignment2a *****\n\n")
    # Code start here

    # Read data from .csv file
    fileName = "Assignment02/tripadvisor_review.csv"
    labels, data = readCSV(fileName)

    # Randomly split 75% data for training, then the rest 25% data for testing
    trainingData_indexes = np.random.choice(data.shape[1], size=round(data.shape[1] * 0.75), replace=False)
    trainingData = data[:, trainingData_indexes]
    trainingLabels = labels[:, trainingData_indexes]
    testingData_indexes = np.delete(np.arange(data.shape[1]), trainingData_indexes)
    testingData = data[:, testingData_indexes]
    testingLabels = labels[:, testingData_indexes]

    a = kNNClassifier(trainingData, trainingLabels, testingData, 3)
    print(a)


if __name__ == '__main__':
    """ Execute main() function """
    pass

    main()
