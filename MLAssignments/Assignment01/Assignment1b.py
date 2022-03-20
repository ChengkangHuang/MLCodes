""" This file describes code assignment1b for COMP10200.
    Chengkang Huang, 000787205, Mohawk College, 2022
"""

# Import the libraries
import csv
import numpy as np
from matplotlib import pyplot as plt
from enum import Enum


class Category(Enum):
    """ Enum all (Category) feature names """
    Category1 = "Art_galleries"
    Category2 = "Dance_clubs"
    Category3 = "Juice_bars"
    Category4 = "Restaurants"
    Category5 = "Museums"
    Category6 = "Resorts"
    Category7 = "Picnic_spots"
    Category8 = "Beaches"
    Category9 = "Theaters"
    Category10 = "Religious_institutions"


class Rating(Enum):
    """ Enum rating labels """
    Terrible = 0
    Poor = 1
    Average = 2
    VeryGood = 3
    Excellent = 4


def readCSV(fileName: str):
    """ Read the .csv file and combine all data into a large dataset
    
    Read .csv file from current folder, append each record data into each
    feature name array, finally combine them into a new large numpy array.

    Arg:
        fileName(str): The string that contain the .csv file location.

    Return:
        A new large numpy array that contain all record from .csv file.
        Ex:
        np.array([UserId, Art_galleries, Dance_clubs, Juice_bars, Restaurants, Museums, Resorts, Picnic_spots, Beaches, Theaters, Religious_institutions])

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

    try:
        with open(fileName) as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    UserId.append(int(row['UserID']))
                    Art_galleries.append(float(row[Category.Category1.value]))
                    Dance_clubs.append(float(row[Category.Category2.value]))
                    Juice_bars.append(float(row[Category.Category3.value]))
                    Restaurants.append(float(row[Category.Category4.value]))
                    Museums.append(float(row[Category.Category5.value]))
                    Resorts.append(float(row[Category.Category6.value]))
                    Picnic_spots.append(float(row[Category.Category7.value]))
                    Beaches.append(float(row[Category.Category8.value]))
                    Theaters.append(float(row[Category.Category9.value]))
                    Religious_institutions.append(float(row[Category.Category10.value]))
                except:
                    print("Read row error")
    except:
        print("Open file error")

    return np.array([UserId, Art_galleries, Dance_clubs, Juice_bars, Restaurants, Museums, Resorts, Picnic_spots, Beaches, Theaters, Religious_institutions])


def evaluation(score: int) -> str:
    """ Check the score from the data and give back the label
    
    Base on the rating score, compare with the enum "Rating" to set
    the label for each record.

    Arg:
        score(int): The number from each record.

    Return:
        resultStr(str): The label base on different input.

    """
    pass

    resultStr = ""
    if score == Rating.Terrible.value:
        resultStr = "Terrible"
    elif score == Rating.Poor.value:
        resultStr = "Poor"
    elif score == Rating.Average.value:
        resultStr = "Average"
    elif score == Rating.VeryGood.value:
        resultStr = "VeryGood"
    elif score == Rating.Excellent.value:
        resultStr = "Excellent"
    else:
        resultStr = "No rating scores"
    return resultStr


def summarizeData(category: Category, dataset: np.array):
    """ Print the summarize data for each feature
    
    Format the print output for each feature which need to be summarized.

    Arg:
        category(Category): Feature name from enum option.
        dataset(np.array): The numpy array that need to process.

    Return:
        Print the summarize data by the formatter.
    """
    minNum = np.min(dataset)
    maxNum = np.max(dataset)
    avrNum = np.round(np.mean(dataset), 2)
    print(
        """{} summarize => Lowest rating:  {} [{}] | Highest rating: {} [{}] | Average rating: {} [{}]"""
            .format(
                category,
                minNum, evaluation(int(minNum)),
                maxNum, evaluation(int(maxNum)),
                avrNum, evaluation(int(avrNum))
            )
        )


def main():
    """ Main code execute module

    Read the data from .csv file, push the data into right place, split the whole
    dataset to 75% training data, and 25% testing data. Summarize the training data
    and testing, after that output the scatter plot and the bar plot base on the data.
    
    """
    pass

    print("\n\n***** Look in the code for COMP10200 code assignment1b *****\n\n")
    # Code start here
    
    # Read data from .csv file
    fileName = "Assignment01/tripadvisor_review.csv"
    data = readCSV(fileName)

    # Randomly split 75% data for training, then the rest 25% data for testing
    feature_names = np.array([Category.Category1.value, Category.Category2.value, Category.Category3.value, 
                                Category.Category3.value, Category.Category5.value, Category.Category6.value,
                                Category.Category7.value, Category.Category8.value, Category.Category9.value,
                                Category.Category10.value])
    trainingData_indexes = np.random.choice(data.shape[1], size = round(data.shape[1] * 0.75), replace = False)
    trainingData = data[:, trainingData_indexes]
    labels = np.array([Rating.Terrible.name, Rating.Poor.name, Rating.Average.name, Rating.VeryGood.name, Rating.Excellent.name])
    testingData_indexes = np.delete(np.arange(data.shape[1]), trainingData_indexes)
    testingData = data[:, testingData_indexes]
    
    # Summarize training data
    print("************************************ Training Data ************************************")
    summarizeData(feature_names[0], trainingData[1])    # Art_galleries
    summarizeData(feature_names[1], trainingData[2])    # Dance_clubs
    summarizeData(feature_names[2], trainingData[3])    # Juice_bars
    summarizeData(feature_names[3], trainingData[4])    # Restaurants
    summarizeData(feature_names[4], trainingData[5])    # Museums
    summarizeData(feature_names[5], trainingData[6])    # Resorts
    summarizeData(feature_names[6], trainingData[7])    # Picnic_spots
    summarizeData(feature_names[7], trainingData[8])    # Beaches
    summarizeData(feature_names[8], trainingData[9])    # Theaters
    summarizeData(feature_names[9], trainingData[10])   # Religious_institutions

    # Summarize testing data
    print("\n************************************ Testing Data ************************************")
    summarizeData(feature_names[0], testingData[1])    # Art_galleries
    summarizeData(feature_names[1], testingData[2])    # Dance_clubs
    summarizeData(feature_names[2], testingData[3])    # Juice_bars
    summarizeData(feature_names[3], testingData[4])    # Restaurants
    summarizeData(feature_names[4], testingData[5])    # Museums
    summarizeData(feature_names[5], testingData[6])    # Resorts
    summarizeData(feature_names[6], testingData[7])    # Picnic_spots
    summarizeData(feature_names[7], testingData[8])    # Beaches
    summarizeData(feature_names[8], testingData[9])    # Theaters
    summarizeData(feature_names[9], testingData[10])   # Religious_institutions

    # Draw the graph
    # Scatter plot 1
    plt.figure(1)
    plt.title("Art_galleries vs Dance_clubs")
    plt.xlabel(Category.Category1.value)
    plt.ylabel(Category.Category2.value)
    plt.scatter(trainingData[0], trainingData[1], marker=".", c="pink")
    plt.scatter(trainingData[0], trainingData[2], marker=".", c="purple")
    plt.show()

    # Scatter plot 2
    plt.figure(2)
    plt.title("Juice_bars vs Restaurants")
    plt.xlabel(Category.Category3.value)
    plt.ylabel(Category.Category4.value)
    plt.scatter(trainingData[0], trainingData[3], marker=".", c="blue")
    plt.scatter(trainingData[0], trainingData[4], marker=".", c="green")
    plt.show()
    
    # Scatter plot 3
    plt.figure(3)
    plt.title("Museums vs Resorts")
    plt.xlabel(Category.Category5.value)
    plt.ylabel(Category.Category6.value)
    plt.scatter(trainingData[0], trainingData[5], marker=".", c="red")
    plt.scatter(trainingData[0], trainingData[6], marker=".", c="orange")
    plt.show()

    # Bar plot
    # plt.figure(4)
    # plt.title("Frequency of each label")
    # plt.xlabel("Rating Labels")
    # plt.ylabel("Rating Socres")
    # plt.bar(labels, trainingData[1:, 1])
    # plt.grid(True)
    # plt.show()


if __name__ == '__main__':
    """ Execute main() function """
    pass

    main()
