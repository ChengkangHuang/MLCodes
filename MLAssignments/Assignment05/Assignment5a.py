""" This file describes code assignment5a for COMP10200.
    Chengkang Huang, 000787205, Mohawk College, 2022
"""

""" The K means value that I prefer is 8. """

# Import the libraries
import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def k_means_inertia_calculation(datasets: np.ndarray, k: int):
    """ Calculate the inertia of K-means algorithm

    Arg:
        datasets(np.ndarray): The dataset that will be used to calculate the inertia.
        k(int): The number of clusters.

    Return:
        The inertia of K-means algorithm.
    """
    pass

    # Code start here
    kmeans = KMeans(n_clusters=k, random_state=0).fit(datasets)
    return kmeans.inertia_


def main():
    """ Main code execute module """
    pass

    print("\n\n***** Look in the code for COMP10200 code assignment5a *****\n")
    # Code start here

    try:
        # Set the absolute path
        dirPath = os.path.abspath(".")

        print("Task #1 Result:")
        # Please replace the path if needed
        living_room_img = dirPath + "\\MLAssignments\\Assignment05\\" + "living_room.jpg"
        task1_img = io.imread(living_room_img)
        task1_img = task1_img.reshape(task1_img.shape[0] * task1_img.shape[1], 3)
        inertia = []
        for k in range(2, 20, 2):
            SSE = k_means_inertia_calculation(task1_img, k)
            inertia.append(SSE)
            print("k=%s\t SSE = %f" % (k, SSE))
        plt.plot(range(2, 20, 2), inertia)
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.title("Inertia of K-means algorithm")
        plt.show()
        
        print("\nTask #2 Result:")
        stairs_img = dirPath + "\\MLAssignments\\Assignment05\\" + "stairs.jpg"
        task2_img = io.imread(stairs_img)
        shape=task2_img.shape
        new_task2_img = task2_img.reshape(shape[0] * shape[1], 3)
        clusting = KMeans(n_clusters=8, random_state=0).fit(new_task2_img)
        centroids = clusting.cluster_centers_
        cls_pred = clusting.labels_
        for i in range(0, 8):
            new_task2_img[cls_pred == i] = centroids[i]
        new_task2_img = new_task2_img.reshape(shape[0], shape[1], 3)
        plt.title("Clustering result")
        plt.imshow(new_task2_img)

    except:
        print("Program run failed")


if __name__ == '__main__':
    """ Execute main() function """
    pass

    main()
