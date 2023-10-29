from abc import ABC, abstractmethod
import numpy as np
import sympy as sp
from stlpy.STL import (LinearPredicate, NonlinearPredicate)


def flatten_before_sub(formula, K):
    made_modification = False
    if formula.timesteps == list(range(K + 1)):
        return made_modification
    for subformula in formula.subformula_list:
        if subformula.timesteps == list(range(K + 1)):
            pass
        else:
            if formula.combination_type == subformula.combination_type:
                # Remove the subformula
                i = formula.subformula_list.index(subformula)
                made_modification = flatten_before_sub(subformula, K) or made_modification
                formula.subformula_list.pop(i)
                st = formula.timesteps.pop(i)

                # Add all the subformula's subformulas instead
                formula.subformula_list += subformula.subformula_list
                formula.timesteps += [t + st for t in subformula.timesteps]
                made_modification = True
            else:
                made_modification = flatten_before_sub(subformula, K) or made_modification

    return made_modification

class Model(ABC):

    def __init__(self, K, t_f_guess, max_k=5, smin_C=0.1):
        self.K = K
        self.t_f_guess = t_f_guess
        self.max_k = max_k
        self.C = smin_C
        self.known_gradient = dict()
        self.known_lambda = dict()

    def log_min(self, list):
        len_ = len(list)
        if isinstance(list, np.ndarray):
            result = 0
            for i in range(len_):
                result += np.exp(-self.max_k * list[i])
            result = - 1 / self.max_k * np.log(result)
        else:
            result = 0
            for i in range(len_):
                result += sp.exp(-self.max_k * list[i])
            result = - 1 / self.max_k * sp.log(result)
        return result

    def e_max(self, list):
        len_ = len(list)
        numerator = 0
        denominator = 0
        if isinstance(list, np.ndarray):
            result = 0
            for i in range(len_):
                numerator += list[i] * np.exp(self.max_k * list[i])
                denominator += np.exp(self.max_k * list[i])
            result = numerator / denominator
        else:
            result = 0
            for i in range(len_):
                numerator += list[i] * sp.exp(self.max_k * list[i])
                denominator += sp.exp(self.max_k * list[i])
            result = numerator / denominator
        return result

    def flatten_before_sub(self, formula):
        return flatten_before_sub(formula, self.K)

    @abstractmethod
    def calculate_n_x_sub(self, spec, flag): ## flag = 0 before "always", flag = 1 after "always"
        pass

    @abstractmethod
    def add_interpolate_function(self, spec, index, var):  ### flag = 0 before temp logical flag=1 after
        pass

    @abstractmethod
    def get_equations(self):
        """
        :return: Functions to calculate A, B and f given state x and input u
        """
        pass

    @abstractmethod
    def initialize_trajectory(self, X, X_sub, X_robust, U):
        """
        Initialize the trajectory.

        :param X: Numpy array of states to be initialized
        :param U: Numpy array of inputs to be initialized
        :return: The initialized X and U
        """

        pass

    @abstractmethod
    def calculate_x_sub(self, spec, index, state, X_sub, X_robust):
        pass

    @abstractmethod
    def get_objective(self, X_v, X_robust_v, U_v, X_last_p, X_robust_last_p, U_last_p):
        """
        Get model specific objective to be minimized.

        :param X_v: cvx variable for current states
        :param U_v: cvx variable for current inputs
        :param X_last_p: cvx parameter for last states
        :param U_last_p: cvx parameter for last inputs
        :return: A cvx objective function.
        """
        pass

    @abstractmethod
    def get_constraints(self, X_v, X_robust_v, U_v, X_last_p, X_robust_last_p, U_last_p):
        pass

    @abstractmethod
    def add_sub_dynamics_constrains(self, X_v, X_sub_v, X_robust_v, U_v, X_last_p,
                                    X_sub_last_p, X_robust_last_p, U_last_p, nu_sub, nu_robust,
                                    par_sub_dynamics_matrix):
        pass

    @abstractmethod
    def get_sub_dynamics_constrains(self, X_last_p,
                                    X_sub_last_p, X_robust_last_p, U_last_p, par_sub_dynamics_matrix):
        pass

    @abstractmethod
    def get_sub_dynamics_constrains_parameters(self, input_value, var, var_index, var_lambda_index, i):
        pass

    @abstractmethod
    def get_linear_cost(self):
        """
        :return: linearized nonlinear constrains cost (flexible constrain variable)
        """
        pass

    @abstractmethod
    def get_nonlinear_cost(self, X=None, U=None):
        """
        :return: nonlinear constrains cost
        """
        pass

    @abstractmethod
    def calculate_subdynamics(self, X_nl):
        pass

    def smin(self, a_func, b_func, C, input_value, input_variable,
             non_center_var=None):
        value_func = sp.zeros(1, 1)
        gradient_func = None

        a_lamdba = self.f_lambdify((input_variable, ), a_func)
        b_lamdba = self.f_lambdify((input_variable, ), b_func)
        if a_lamdba(input_value) - b_lamdba(input_value) <= -C:
            value_func[0, 0] = a_func
        elif a_lamdba(input_value) - b_lamdba(input_value) >= C:
            value_func[0, 0] = b_func
        else:
            h = sp.zeros(1, 1)
            g_func = sp.zeros(1, 1)
            h[0, 0] = 1 / 2 + (a_func - b_func) / 2 / C
            g_func[0, 0] = a_func * (1 - h[0, 0]) + h[0, 0] * b_func # - C * h[0, 0] * (1 - h[0, 0])
            value_func[0, 0] = g_func
        if non_center_var is None:
            gradient_func = self.cal_f_gradient(input_variable, value_func)
        else:
            gradient_func = self.cal_f_gradient(non_center_var, value_func)
        return value_func, gradient_func

    def f_lambdify(self, var, f_):
        f_func = None
        if isinstance(f_, sp.MutableDenseMatrix):
            if (str(var[0]), f_[0, 0]) not in self.known_lambda:
                f_func = sp.lambdify(var, f_, 'numpy')
                self.known_lambda[(str(var[0]), f_[0, 0])] = f_func
            else:
                f_func = self.known_lambda[(str(var[0]), f_[0, 0])]
        else:
        #elif isinstance(f_, sp.Symbol):
            if (str(var[0]), f_) not in self.known_lambda:
                f_func = sp.lambdify(var, f_, 'numpy')
                self.known_lambda[(str(var[0]), f_)] = f_func
            else:
                f_func = self.known_lambda[(str(var[0]), f_)]
        return f_func

    def cal_f_gradient(self, var, f_):

        if f_[0, 0] not in self.known_gradient:
            f_gradient = f_.jacobian(var)
            self.known_gradient[f_[0, 0]] = f_gradient
        else:
            f_gradient = self.known_gradient[f_[0, 0]]
        return f_gradient


class MovingNonlinearPredicate(NonlinearPredicate):
    def __init__(self, g, d, center, name=None):
        NonlinearPredicate.__init__(self, g, d, name)
        self.center_position = center
    def negation(self):
        if self.name is None:
            newname = None
        else:
            newname = "not " + self.name

        negative_g = lambda y : -self.g(y)
        return MovingNonlinearPredicate(negative_g, self.d, self.center_position, name=newname)



def inside_moving_circle_formula(center_position, radius, y1_index, y2_index, center_index_1, center_index_2, d, name=None):
    def g(y):
        y1 = y[y1_index]
        y2 = y[y2_index]
        return radius**2 - (y1-y[center_index_1])**2 - (y2-y[center_index_2])**2
    return MovingNonlinearPredicate(g, d, center_position, name=name)

