# CS445 Program 3
# Christopher Juncker


import numpy as np


# "Assignment #1: K-Means
# "  Implement the standard version of the K-Means algorithm as described in lecture. The initial
# "  starting points for the K cluster means can be K randomly selected data points. You should
# "  have an option to run the algorithm r times from r different randomly chosen initializations
# "  (e.g., r = 10), where you then select the solution that gives the lowest sum of squares error
# "  over the r runs. Run the algorithm for several different values of K and report the sum of
# "  squares error for each of these models. Please include a 2-d plot of several different
# "  iterations of your algorithm with the data points and clusters.
# "
class KMeans:
    def __init__(self):
        self.data = np.loadtxt("./445_cluster_dataset.txt")




def main():
    print("Program 3")

    k = KMeans()

    print(k.data)
    print(len(k.data))


if __name__ == '__main__':
    main()
