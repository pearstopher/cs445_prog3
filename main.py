# CS445 Program 3
# Christopher Juncker
#
# Fuzzy C-Means


import numpy as np
import math
import matplotlib.pyplot as plt

# " m > 1 is a 'fuzzifier' parameter (fix this value during the algorithm)
M = 1.25


# "Assignment #2: Fuzzy C-Means
# "  Implement the standard version of the fuzzy c-means (FCM) algorithm as described in lecture. As
# "  shown in lecture, the update formulae for the centroids and membership weights are as follows:
# "
# "    c_k = [ sum_x ( w_k (x)^m x ) ] / [ sum_x ( w_k (x)^m ) ]
# "
# "    w_ij = 1 / [ sum_(k=1)^c ( ||x_i - c_j|| / ||x_i - c_k|| )^(2/(m-1)) ]
# "
# "  Where m > 1 is a "fuzzifier" parameter (just fix this value during the algorithm – you are welcome to
# "  experiment by trying different values for different runs of FCM if you wish). Begin by initializing the
# "  centroids randomly, then compute the weights, update the centroids, recompute the weights, etc. As
# "  before, you should have an option to run the algorithm r times from r different randomly chosen
# "  initializations (e.g., r = 10), where you then select the solution that gives the lowest sum of squares
# "  error over the r runs. Run the algorithm for several different values of K (where K is the number of
# "  clusters) and report the sum of squares error for each of these models. Please include a 2-d plot of
# "  several different iterations of your algorithm with the data points and clusters.
#
class CMeans:
    def __init__(self, k, trial=0, display_plot=False, save_image=False):
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
        self.save_image = save_image
        self.trial = trial

    # "Basic Fuzzy C-Means algorithm
    # "  1. Choose a number of clusters: C
    # "  2. Assign coefficients randomly to each data point for being
    # "     in the clusters (these are the initial membership grades).
    # "  3. repeat:
    # "  4.   compute the centroid for each cluster (m-step)
    # "  5.   for each data point, compute its coefficients/membership
    # "       grades for being in the clusters (e-step).
    #
    # W - membership grades
    # w_ij - i = 1-n, j=1-c
    # m - fuzzifier
    # c - cluster
    def run(self):
        count = 0

        # "Assign coefficients randomly to each data point for being
        # "in the clusters
        #
        # create k random values in the range [0,1) for each data point
        # 0 = not a member, 1 = a member
        membership_grades = [np.random.random_sample((self.k, 0)) for _ in self.data]

        # hold the previous centers in a separate array
        # (to compare against, for stopping condition)
        self.old_points = np.empty((self.k, 2))

        # repeat (until centroid location points do not change)
        while not np.array_equal(self.old_points, self.points):
            count += 1
            # display and formatting
            print("Loop", count, end=" ")
            if count % 10 == 0:
                print("\n\t\t", end="")

            # save the cluster locations before updating them
            self.old_points = np.copy(self.points)

            # "Compute the centroid for each cluster (m-step)
            self.points = np.empty((self.k, 2))
            # for each cluster center point,
            for i, p in enumerate(self.points):
                numerator = 0
                denominator = 0
                # loop through all of the data
                for j, d in enumerate(self.data):
                    # and calculate the sums for the equation
                    denominator += membership_grades[j][i] ** M
                    numerator = denominator * d
                # calculate the cluster center
                p = numerator / denominator
                # and set it in the original array (p is local)
                self.points[i] = p

            # "for each data point, compute its coefficients/membership
            # "grades for being in the clusters (e-step).
            #
            # loop through each data item
            for i, d in enumerate(self.data):
                # loop through each of the clusters
                for j, c in enumerate(self.points):
                    # compute the new membership grade

                    # compute the sum
                    sigma = 0
                    # (loop through clusters again)
                    for k, ck in enumerate(self.points):
                        a = d-c
                        b = d-ck
                        # sigma += self.l2(d - c) / self.l2(d - ck)
                        sigma = math.sqrt(a[0]**2 + a[1]**2) / (math.sqrt(b[0]**2 + b[1]**2) + 0.0000001)
                        if sigma == 0:
                            sigma = 0.00001  # division by 0
                    # calculate the denominator
                    denominator = sigma ** (2 / (M - 1))
                    # complete the equation
                    membership_grades[i][j] = 1 / denominator

            # convert the membership grades to clusters to display
            # (highest grade = current cluster membership)
            self.clusters = [[] for _ in range(self.k)]
            for i, d in enumerate(self.data):
                self.clusters[np.argmax(membership_grades[i])].append(d)

            # plot the updated points
            if self.display_plot:
                self.plot(self.trial, count, self.save_image)

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
    def plot(self, trial, run, save_image):
        # if this is a set of trials,
        # only make graphs for the first trial in the set
        if trial != 0:
            return

        for p, c, color in zip(self.points, self.clusters,
                               iter(plt.cm.rainbow(np.linspace(0, 1, len(self.points))))):
            np.array(c).reshape(2, -1)
            for a in c:
                plt.scatter(a[0], a[1], color=color)
            plt.scatter(p[0], p[1], c='black')

        plt.xlim([-5, 5])
        plt.ylim([-5, 5])

        if save_image:
            plt.savefig("./images/k" + str(self.k) + "_run" + str(run) + ".png")
        else:
            plt.show()

        plt.clf()


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
            k = CMeans(k_values[i], j, display_plot=True, save_image=True)
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
