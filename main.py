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
    def __init__(self, num_points):
        # load the data
        self.data = np.loadtxt("./445_cluster_dataset.txt")
        # self.data = self.data[0:750]  # use less data to save time testing
        # self.data_x, self.data_y = np.split(self.data, [-1], axis=1)  # Or simply : np.split(Xy,[-1],1)
        # print(self.data_x)
        # print(self.data_y)

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
        points = np.random.random_sample((self.num_points, 2))
        points -= 0.5  # center at 0
        old_points = np.empty((self.num_points, 2))

        # "repeat until Centroids do not change.
        while not np.array_equal(old_points, points):
            count += 1
            print("Num loops:", count)

            # "Form K clusters by assigning each point to its closest centroid.
            clusters = [[] for _ in range(self.num_points)]
            # clusters = [np.zeros(0) for _ in range(self.num_points)]
            for d in self.data:
                distances = [self.l2(points[i], d) for i in range(self.num_points)]
                clusters[np.argmin(distances)].append(d)
                # clusters[np.argmin(distances)] = np.append(clusters[np.argmin(distances)], d)

            # plot the updated points
            self.plot(points, clusters)

            # "Recompute the centroid of each cluster
            old_points = np.copy(points)
            for i in range(self.num_points):
                array_x = np.empty(len(clusters[i]))
                array_y = np.empty(len(clusters[i]))
                for j in range(len(clusters[i])):
                    array_x[j] = clusters[i][j][0]
                    array_y[j] = clusters[i][j][1]

                # points[i][0] = np.mean(clusters[i][0])
                # points[i][1] = np.mean(clusters[i][1])
                points[i][0] = np.mean(array_x)
                points[i][1] = np.mean(array_y)



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

    k = KMeans(10)

    k.run()


if __name__ == '__main__':
    main()
