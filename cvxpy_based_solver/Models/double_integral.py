from .base_model import Model

import sympy as sp
import numpy as np
import cvxpy as cvx


class DoubleIntegral:
    n_x = 4
    n_u = 2

    x_ob = np.array([3.0, 2.0])
    R_0 = 1.0

    def __init__(self, K, t_f_guess):
        """
        A large r_scale for a small scale problem will
        ead to numerical problems as parameters become excessively small
        and (it seems) precision is lost in the dynamics.
        """

        self.K = K
        self.t_f_guess = t_f_guess
        self.x_init = np.array([4.0, 4.0, 0.0, 0.0])
        self.x_final = np.array([0.0, 0.0, 0.0, 0.0])

        # slack variable for linear constraint relaxation
        self.s_prime = cvx.Variable((K, 1), nonneg=True)

        # slack variable for lossless convexification
        # self.gamma = cvx.Variable(K, nonneg=True)

    def get_equations(self):
        """
        :return: Functions to calculate A, B and f given state x and input u
        """
        f = sp.zeros(4, 1)

        x = sp.Matrix(sp.symbols('x y vx vy', real=True))
        u = sp.Matrix(sp.symbols('ax ay', real=True))

        I = np.eye(2)
        O_ = np.zeros((2, 2))

        f[0:2, 0] = x[2:4, 0]
        f[2:4, 0] = u[0:2, 0]

        f = sp.simplify(f)
        A = sp.simplify(f.jacobian(x))
        B = sp.simplify(f.jacobian(u))

        f_func = sp.lambdify((x, u), f, 'numpy')
        A_func = sp.lambdify((x, u), A, 'numpy')
        B_func = sp.lambdify((x, u), B, 'numpy')

        return f_func, A_func, B_func

    def initialize_trajectory(self, X, U):
        """
        Initialize the trajectory.

        :param X: Numpy array of states to be initialized
        :param U: Numpy array of inputs to be initialized
        :return: The initialized X and U
        """

        for k in range(self.K):
            alpha1 = (self.K - 1 - k) / (self.K - 1)
            alpha2 = k / (self.K - 1)

            X[:, k] = alpha1 * self.x_init + alpha2 * self.x_final

        U = np.zeros((self.n_u, self.K))
        return X, U

    def get_objective(self, X_v, U_v, X_last_p, U_last_p):
        """
        Get model specific objective to be minimized.

        :param X_v: cvx variable for current states
        :param U_v: cvx variable for current inputs
        :param X_last_p: cvx parameter for last states
        :param U_last_p: cvx parameter for last inputs
        :return: A cvx objective function.
        """
        return cvx.Minimize(1e5 * cvx.sum(self.s_prime) + 1e-5 * cvx.sum_squares(U_v))

    def get_constraints(self, X_v, U_v, X_last_p, U_last_p):
        """
        Get model specific constraints.

        :param X_v: cvx variable for current states
        :param U_v: cvx variable for current inputs
        :param X_last_p: cvx parameter for last states
        :param U_last_p: cvx parameter for last inputs
        :return: A list of cvx constraints
        """
        # Boundary conditions:
        constraints = [
            X_v[:, 0] == self.x_init,
            X_v[:, self.K - 1] == self.x_final
        ]

        constraints += [
            # State constraints:
            X_v[0:2, :] <= 6, X_v[0:2, :] >= -1,
            X_v[2:4, :] <= 1, X_v[2:4, :] >= -1,
            U_v[:, :] <= 1, U_v[:, :] >= -1
        ]

        # linearized lower thrust constraint
        for k in range(self.K):
            constraints += [
                -(X_last_p[1, k] - self.x_ob[1]) ** 2 - (X_last_p[0, k] - self.x_ob[0]) ** 2 - 2 * (
                        X_last_p[0, k] - self.x_ob[0]) * (X_v[0, k] - X_last_p[0, k]) - 2 * (
                        X_last_p[1, k] - self.x_ob[1]) * (X_v[1, k] - X_last_p[1, k]) + self.R_0 ** 2 <= self.s_prime[
                    k, 0]
            ]
        return constraints

    def get_linear_cost(self):
        cost = np.sum(self.s_prime.value)
        return cost

    def get_nonlinear_cost(self, X=None, U=None):
        magnitude = (X[1, :] - self.x_ob[1]) ** 2 + (X[0, :] - self.x_ob[0]) ** 2
        is_violated = magnitude < self.R_0 ** 2
        violation = self.R_0 ** 2 - magnitude
        cost = np.sum(is_violated * violation)
        return cost
