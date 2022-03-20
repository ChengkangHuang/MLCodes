""" This file describes code assignment1a for COMP10200.
    Chengkang Huang, 000787205, Mohawk College, 2022
"""

# Import the libraries
import numpy as np
import time as t


def numInputHandler(msg: str):
    """ Handle the number input for user

    Receives string input from user, check the string value if it's available to
    convert to integer value.

    Args:
        msg(str): The string that show up in the console to mention what user should input before user start typing.

    Returns:
        inputNum(int): The integer value which is validated by error handler

    Raises:
        Type Error: Stay in the while loop until receive the correct value

    """
    pass

    inputNum = 0
    while (True):
        try:
            inputNum = int(input(msg))
            break
        except:
            print("Please enter a number")
    return inputNum


def nameInputHandler(players: int):
    """ Handle the player name input for user

    Reveices string input from user, ignore all " "(space) and null input,
    and make sure the string value in nameSet is unique.

    Args:
        players(int): The integer value that use for create the limited value in nameSet.

    Returns:
        nameSet(numpy.array): Convert the nameSet to numpy array, in order to do the sorting in other steps
        
    """
    pass

    print("\nEnter " + str(players) + " player names")
    nameSet = set(())
    while players > 0:
        name = input().replace(" ", "").capitalize()
        if name != "":
            if name not in nameSet:
                nameSet.add(name)
                players -= 1
            else:
                print("Name already entered")
                continue
        else:
            print("Name can't be null")
            continue
    return np.array(list(nameSet))


def enterInputHandler(nameCollection: np.array, players: int, tIntervals: int):
    """ Handle the "enter" key pressed for user

    Records the interval time between user pressed the "enter" key, and store them
    into 2-D numpy array

    Args:
        nameCollection(numpy.array): Switch to each player's turn base on nameCollection
        players(int): A row number for create the 2-D numpy array
        tIntervals(int): A column number for create the 2-D numpy array

    Returns:
        timeCollection(numpy.array): 2-D numpy array include all interval times for each player

    """
    pass

    timeCollection = np.zeros((players, tIntervals))
    for i in range(len(nameCollection)):
        print("\n" + nameCollection[i] + "'s turn. Press enter " +
              str(tIntervals + 1) + " times quickly.")
        count = 0
        tList = np.zeros((tIntervals), dtype=float)
        input("Press enter to start...")
        while count < tIntervals:
            startTime = t.time()
            input()
            endTime = t.time()
            tList[count] = round((endTime - startTime), 3)
            count += 1
        timeCollection[i] = tList
    return timeCollection


def main():
    """ Main code execute module.

    Receives users input, and creates numpy array matrix base on users input.
    Collects all interval times between two "enter" key had been pressed,
    and store time into data collection.
    Format and output the result base on the data collection with using numpy built-in functions

    """
    pass

    print("\n\n***** Look in the code for COMP10200 code assignment1a *****\n\n")

    # Receive the input from user
    players = numInputHandler("How many players? ")
    tIntervals = numInputHandler("How many time intervals? ")
    names = nameInputHandler(players)
    times = enterInputHandler(names, players, tIntervals)

    # Output result
    print("Names: {}".format(names[names.argsort()]))
    print("Mean times: {}".format(
        times[names.argsort()].mean(axis=1).round(3)))
    print("Fastest Average Time: {0:.3f}s by {1}".format(
        np.mean(times, axis=1).min(), names[np.average(times, axis=1).argmin()]))
    print("Slowest Average Time: {0:.3f}s by {1}".format(
        np.mean(times, axis=1).max(), names[np.average(times, axis=1).argmax()]))
    print("Fastest Single Time: {0:.3f}s by {1}".format(
        np.min(times, axis=1).min(), names[np.min(times, axis=1).argmin()]))
    print("Slowest Single Time: {0:.3f}s by {1}".format(
        np.max(times, axis=1).max(), names[np.max(times, axis=1).argmax()]))

    # Raw data output
    print("\nRaw data output:")
    print(names)
    print(times)


if __name__ == '__main__':
    """ Execute main() function """
    pass

    main()
