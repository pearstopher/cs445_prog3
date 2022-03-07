# CS445 Program 3
# Christopher Juncker


import numpy as np
import math


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
    def __init__(self, num_points):
        # load the data
        self.data = np.loadtxt("./445_cluster_dataset.txt")

        # save the number of points
        self.num_points = num_points

    # "Basic K-Means algorithm
    # "  1. Select K points as initial centroids.
    # "  2. repeat
    # "  3.   Form K clusters by assigning each point to its closest centroid.
    # "  4.   Recompute the centroid of each cluster.
    # "  5. until Centroids do not change.
    def run(self):
        count = 0

        # "Select K points as initial centroids
        points = np.random.rand(self.num_points, 2)
        old_points = np.empty((self.num_points, 2))

        # "repeat until Centroids do not change.
        while not np.array_equal(old_points, points):
            count += 1
            print("Num loops:", count)

            # "Form K clusters by assigning each point to its closest centroid.
            clusters = [[] for _ in range(self.num_points)]
            for d in self.data:
                distances = [self.l2(points[i], d) for i in range(self.num_points)]
                clusters[np.argmin(distances)].append(d)

            # "Recompute the centroid of each cluster
            old_points = np.copy(points)
            for i in range(self.num_points):
                points[i] = np.mean(clusters[i])

        return

    # "For numerical attributes, often use L2 (Euclidean) distance:
    # "
    # "  d(x, y) = sqrt( sum{i=1, n} (xi - yi)^2 )
    @staticmethod
    def l2(x, y):
        total = 0
        for xi, yi in zip(x, y):
            total += (xi - yi)**2
        total = math.sqrt(total)
        return total


def main():
    print("Program 3")

    k = KMeans(5)

    k.run()


if __name__ == '__main__':
    main()
