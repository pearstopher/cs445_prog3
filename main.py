# CS445 Program 3
# Christopher Juncker


import numpy as np
import math
import matplotlib.pyplot as plt


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
    def __init__(self, k, display_plot=False):
        # load the data
        self.data = np.loadtxt("./445_cluster_dataset.txt")
        # save the number of points
        self.k = k
        # variables to hold points and clusters
        self.points = []
        self.old_points = []
        self.clusters = []
        # print a graph or not?
        self.display_plot = display_plot

    # "Basic K-Means algorithm
    # "  1. Select K points as initial centroids.
    # "  2. repeat
    # "  3.   Form K clusters by assigning each point to its closest centroid.
    # "  4.   Recompute the centroid of each cluster.
    # "  5. until Centroids do not change.
    def run(self):
        count = 0

        # "Select K points as initial centroids
        self.points = np.random.random_sample((self.k, 2))
        self.points -= 0.5  # center at 0
        self.old_points = np.empty((self.k, 2))

        # "repeat until Centroids do not change.
        while not np.array_equal(self.old_points, self.points):
            count += 1
            # display and formatting
            print("Loop", count, end=" ")
            if count % 10 == 0:
                print("\n\t\t", end="")

            # "Form K clusters by assigning each point to its closest centroid.
            self.clusters = [[] for _ in range(self.k)]
            for d in self.data:
                distances = [self.l2(self.points[i], d) for i in range(self.k)]
                self.clusters[np.argmin(distances)].append(d)

            # plot the updated points
            if self.display_plot:
                self.plot(self.points, self.clusters)

            # "Recompute the centroid of each cluster
            self.old_points = np.copy(self.points)
            for i in range(self.k):
                array_x = np.empty(len(self.clusters[i]))
                array_y = np.empty(len(self.clusters[i]))
                for j in range(len(self.clusters[i])):
                    array_x[j] = self.clusters[i][j][0]
                    array_y[j] = self.clusters[i][j][1]

                # points[i][0] = np.mean(clusters[i][0])
                # points[i][1] = np.mean(clusters[i][1])
                self.points[i][0] = np.mean(array_x)
                self.points[i][1] = np.mean(array_y)

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

    # "report the sum of squares error for each of these models
    def error(self):
        error = 0
        for point, cluster in zip(self.points, self.clusters):
            for c in cluster:
                error += (point[0] - c[0])**2
                error += (point[1] - c[1])**2
        return error

    # for the graphs
    @staticmethod
    def plot(points, clusters):
        for p, c, color in zip(points, clusters,
                               iter(plt.cm.rainbow(np.linspace(0, 1, len(points))))):
            np.array(c).reshape(2, -1)
            for a in c:
                plt.scatter(a[0], a[1], color=color)
            # print(c)
            # plt.scatter(c[:0], c[:1], c=color)
            plt.scatter(p[0], p[1], c='black')

        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.show()


def main():
    print("Program 3")

    # "You should have an option to run the algorithm r times
    # "  from r different randomly chosen initializations
    # "  (e.g., r = 10)
    r = 10

    # "select the solution that gives the lowest sum of squares error
    # "  over the r runs
    error = np.empty(r)

    # "Run the algorithm for several different values of K and report
    # "  the sum of squares error for each of these models.
    k_values = (2, 3, 5, 10)
    k_error = np.empty(4)

    for i in range(len(k_values)):
        print("\nK-value:", k_values[i])

        for j in range(r):
            print("\tRun:", j, "\n\t\t", end="")
            k = KMeans(k_values[i])
            k.run()
            error[j] = k.error()
            print("\n\t\tError:", error[j])

        k_error[i] = np.amin(error)
        print("\tMinimum error for K-value " + str(k_values[i]) + ": " + str(k_error[i]))

    # I guess the error just gets smaller as K gets bigger, no big surprise
    print("\n\nMinimum error value:", np.amin(k_error))
    print("This corresponds to a K-value of", k_values[int(np.argmin(k_error))])
    print("(K, RSS)")
    for z in zip(k_values, k_error):
        print(z)


if __name__ == '__main__':
    main()
