import cvxpy as cp
import numpy as np
import matplotlib.pyplot as py


class pb_aeroglisseur:
    """Thanks to this class, we can resolve the optimization problem of hovercraft."""

    def __init__(self, k, T, coord1, coord2, rep, l):
        """
        Class initialization

        :param k: number of waypoints
        :type k: entier

        :param T: instants t hit by the hovercraft (coord1, coord2)
        :type T: list of length k

        :param coord1: coordinate number 1 of the point (coord1, coord2) through which the hovercraft passes at time t
        :type coord1: list of length k

        :param coord2: coordinate number 1 of the point (coord1, coord2) through which the hovercraft passes at time t
        :type coord2: list of length k

        :param rep: ndication on the method of resolution used (if rep = 1 lambda = 0 otherwise lambda is to be chosen)
        :type rep: int 1 or 2

        :param l: lambda value, i.e. sensitivity on tradeoff between minimizing u and reaching the crossing points
         if rep = 1, lambda will be equal to 0
        :type l: real positive
        """

        self.k = k
        self.T = T
        self.coord1 = coord1
        self.coord2 = coord2

        if rep == 1:
            self.l = 0
        else:
            self.l = l

        self.wp = np.zeros(k)

        # Associated with this vector, we recover the instant at which the hovercraft reached the last waypoint
        # which tells us the total duration of its journey and which allows us to define the sizes of the vectors
        # u, v, and x
        self.tf = self.T[k-1]

        # We create the summary table of the waypoints
        self.wp = np.array([self.coord1, self.coord2])

       # We create the variables u, v and x with cp.variable()
        self.u = cp.Variable(shape=(2, self.tf))  # fuel (travel cost)
        self.v = cp.Variable(shape=(2, self.tf))  # velocity
        self.x = cp.Variable(shape=(2, self.tf))  # position

        # We define all the constraints according to the method used
        self.contraintes = [self.x[:, self.T[0]] == self.wp[:, 0], self.v[:, 1] == [
            0, 0], ]  # we create the constraint in 0

        for t in range(self.tf-1):  # we make sure that the system is respected
            self.contraintes += [
                self.x[:, t+1] == self.x[:, t] + self.v[:, t],
                self.v[:, t+1] == self.v[:, t] + self.u[:, t],
            ]

        # We define the constraint with respect to the points through which the hovercraft will pass
        if self.l == 0:  # Si on passe par tous les points
            self.contraintes += [self.x[:, self.T[1]] == self.wp[:, 1],
                                 self.x[:, T[2]] == self.wp[:, 2], self.x[:, T[3]-1] == self.wp[:, 3]]

    def reso_et_visu(self):
        """
        This function allows us to solve and visualize graphically
        the solution of the hovercraft optimization problem.

        :return: She returns the path of the hovercraft
        :rtype: plot
        """

        # We define how we want to optimize the cost of the trip as well as the difference at the points
        if self.l == 0:
            objectives = cp.Minimize(cp.sum_squares(self.u))
        else:
            objectives = cp.Minimize(cp.sum_squares(self.u) + self.l*cp.sum_squares(
                (self.x[:, self.T[0:self.k]-np.ones(self.k, int)] - (self.wp[:, 0:self.k]))))

        # We define the optimization problem
        prob = cp.Problem(objectives, self.contraintes)

        # We resolve problem
        prob.solve()

        # We display the result
        py.plot(self.x.value[0, :], self.x.value[1, :], "b.-", markersize=4)
        py.plot(self.x.value[0, :][self.T[0:self.k-1]],
                self.x.value[1, :][self.T[0:self.k-1]], "b.", markersize="12")
        py.plot(self.wp[0, :], self.wp[1, :], "r.", markersize=12)
        py.axis("equal")
        py.title("Hovercraft trajectory")

        py.show()


# Method 1
aeroglisseur = pb_aeroglisseur(
    4, [1, 20, 50, 60], [1, 4, 6, 1], [1, 3, 0, 1], 1, 0.01)
aeroglisseur.reso_et_visu()

# Method 2
aeroglisseur = pb_aeroglisseur(
    4, [1, 20, 50, 60], [1, 4, 6, 1], [1, 3, 0, 1], 2, 0.01)
aeroglisseur.reso_et_visu()