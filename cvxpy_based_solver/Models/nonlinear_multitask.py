import numpy

from .base_model import Model
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import cvxpy as cvx
import stlpy
from stlpy.benchmarks.common import inside_circle_formula, make_circle_patch
from stlpy.benchmarks.base import BenchmarkScenario
from stlpy.STL import (LinearPredicate, NonlinearPredicate)
from stlpy.benchmarks.common import (inside_rectangle_formula,
                                     outside_rectangle_formula,
                                     make_rectangle_patch)


class DoubleIntegral(Model):
    n_x = 4
    n_x_sub = 0
    n_x_robust = 0
    n_u = 2

    def __init__(self, K, t_f_guess, seed=0):
        """
        A large r_scale for a small scale problem will
        ead to numerical problems as parameters become excessively small
        and (it seems) precision is lost in the dynamics.
        """
        self.K = K
        self.t_f_guess = t_f_guess
        self.max_k = 10

        self.x_init = np.array([0.0, 0.0, 0, 0])
        self.x_final = np.array([8.0, 8.0, 0, 0])

        self.u_min = np.array([-5, -5])
        self.u_max = np.array([5, 5])
        self.x_min = np.array([-5.0, -5.0, -5.0, -5.0])
        self.x_max = np.array([10.0, 10.0, 5.0, 5.0])
        # Set the seed for the random number generator (for reproducability)
        np.random.seed(seed=seed)
        num_obstacles = 2
        num_groups = 2
        targets_per_group = 2
        self.targets_per_group = targets_per_group
        # Set the seed for the random number generator (for reproducability)
        np.random.seed(seed)

        # Create the (randomly generated) set of obstacles
        self.obstacles = []
        for i in range(num_obstacles):
            x = np.random.uniform(0, 9)  # keep within workspace
            y = np.random.uniform(0, 9)
            self.obstacles.append((x, y))

        # Create the (randomly generated) set of targets
        self.targets = []
        for i in range(num_groups):
            target_group = []
            for j in range(targets_per_group):
                x = np.random.uniform(0, 9)
                y = np.random.uniform(0, 9)
                target_group.append((x, y))
            self.targets.append(target_group)

        self.goal = (8.0, 8.0)
        # Specify that we must avoid all obstacles
        obstacle_formulas = []
        for obs in self.obstacles:
            tmp_inside = inside_circle_formula(obs, 1.0, 0, 1, 4)
            obstacle_formulas.append(tmp_inside.negation())
        obstacle_avoidance = obstacle_formulas[0]
        for i in range(1, len(obstacle_formulas)):
            obstacle_avoidance = obstacle_avoidance & obstacle_formulas[i]
        #obstacle_avoidance.simplify()
        # Specify that for each target group, we need to visit at least one
        # of the targets in that group
        target_group_formulas = []
        for target_group in self.targets:
            group_formulas = []
            for target in target_group:
                group_formulas.append(inside_circle_formula(target, 1.0, 0, 1, 4))
            reach_target_group = group_formulas[0]
            for i in range(1, self.targets_per_group):
                reach_target_group = reach_target_group | group_formulas[i]
            target_group_formulas.append(reach_target_group)

        # Put all of the constraints together in one specification
        specification = obstacle_avoidance.always(0, self.K)
        for reach_target_group in target_group_formulas:
            reach_target_group.simplify()
            specification = specification & reach_target_group.eventually(0, self.K)
        goal = inside_circle_formula(self.goal, 1.0, 0, 1, 4)
        specification = specification & goal.eventually(0, self.K)
        self.spec = specification
        self.flatten_before_sub(self.spec)

        self.robustness_index = []
        self.calculate_n_x_sub(self.spec, flag=0)
        self.x_sub_init = np.zeros((self.n_x_sub, 1))
        self.known_gradient = dict()
        self.known_lambda = dict()

        str_variable = ''

        ### no until (now)

        for i in range(self.n_x + 2 * self.n_x_sub):  ### x, x_sub, x_sub_previous
            str_variable += 'x_' + str(i)
            if i != self.n_x + 2 * self.n_x_sub - 1:
                str_variable += ', '

        self.var = sp.Matrix(sp.symbols(str_variable, real=True))

        self.f_a = sp.zeros(self.n_x_sub + self.n_x_robust, 1)
        self.f_b = sp.zeros(self.n_x_sub + self.n_x_robust, 1)

        self.var_index = []
        self.max_min_index = np.zeros((self.n_x_sub + self.n_x_robust, 1))
        self.add_interpolate_function(self.spec, 0, self.var)

        self.C = 0.1

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

    def e_max(self, list, epsilon=0.0):
        len_ = len(list)
        numerator = 0
        denominator = 0
        if isinstance(list, np.ndarray):
            result = 0
            for i in range(len_):
                numerator += list[i] * np.exp(self.max_k * list[i])
                denominator += np.exp(self.max_k * list[i])
            result = numerator / (denominator + epsilon)
        else:
            result = 0
            for i in range(len_):
                numerator += list[i] * sp.exp(self.max_k * list[i])
                denominator += sp.exp(self.max_k * list[i])
            result = numerator / (denominator + epsilon)
        return result

    def flatten_before_sub(self, formula):
        made_modification = False
        if formula.timesteps == list(range(self.K + 1)):
            return made_modification
        for subformula in formula.subformula_list:
            if subformula.timesteps == list(range(self.K + 1)):
                pass
            else:
                if formula.combination_type == subformula.combination_type:
                    # Remove the subformula
                    i = formula.subformula_list.index(subformula)
                    formula.subformula_list.pop(i)
                    st = formula.timesteps.pop(i)

                    # Add all the subformula's subformulas instead
                    formula.subformula_list += subformula.subformula_list
                    formula.timesteps += [t + st for t in subformula.timesteps]
                    made_modification = True

                made_modification = self.flatten_before_sub(subformula) or made_modification

        return made_modification

    def calculate_n_x_sub(self, spec, flag): ## flag = 0 before "always", flag = 1 after "always"
        if isinstance(spec, LinearPredicate) or isinstance(spec, NonlinearPredicate):
            return
        if spec.timesteps == list(range(self.K + 1)):
            self.n_x_sub += 1
            self.calculate_n_x_sub(spec.subformula_list[0], 1)
            return
        if flag == 0:
            self.n_x_robust += 1
        else:
            self.n_x_sub += 1
        for i, subformula in enumerate(spec.subformula_list):
            self.calculate_n_x_sub(subformula, flag)

    def add_interpolate_function(self, spec, index, var):  ### flag = 0 before temp logical flag=1 after
        index_ = index
        if spec.timesteps == list(range(self.K + 1)):
            if isinstance(spec.subformula_list[0], NonlinearPredicate):
                self.var_index.append([0, 1, self.n_x + self.n_x_sub + index_])
                self.f_a[index_, 0] = spec.subformula_list[0].g(var[0:4])
                self.f_b[index_, 0] = var[self.n_x + self.n_x_sub + index_, 0]
                ### TODO now 0,1 are manually setting, later use STL to get it.
            elif isinstance(spec.subformula_list[0], LinearPredicate):
                self.var_index.append([0, 1, self.n_x + self.n_x_sub + index_])
                self.f_a[index_, 0] = spec.subformula_list[0].a.T @ var[0:4] - spec.subformula_list[0].b
                self.f_b[index_, 0] = var[self.n_x + self.n_x_sub + index_, 0]
            else:
                index_ = self.add_interpolate_function(spec.subformula_list[0], index_, var)
                self.f_a[index_, 0] = var[self.n_x + self.n_x_sub + index_, 0] ### previous self
                self.f_b[index_, 0] = var[self.n_x + index_ - 1, 0] ### current last sub state
                self.var_index.append([self.n_x + self.n_x_sub + index_, self.n_x + index_ - 1])
            if spec.combination_type == 'or':  #### smax
                self.max_min_index[index_, 0] = 1
            return index_ + 1

        list_function = []
        list_index = []
        nonlinear_flag = 0
        for i, subformula in enumerate(spec.subformula_list):
            if isinstance(subformula, NonlinearPredicate):
                list_function.append(subformula.g(var[0:4]))
                nonlinear_flag = 1
                if not 0 in list_index:
                    list_index.append(0)
                if not 1 in list_index:
                    list_index.append(1)
            elif isinstance(subformula, LinearPredicate):
                tmp = subformula.a.T @ var[0:4] - subformula.b
                list_function.append(tmp[0])
                if not 0 in list_index:
                    list_index.append(0)
                if not 1 in list_index:
                    list_index.append(1)
            else:
                index_ = self.add_interpolate_function(subformula, index_, var)
                list_function.append(var[self.n_x + index_ - 1, 0]) #############TODO robustness index
                list_index.append(self.n_x + index_ - 1)
        g = 0
        if spec.combination_type == 'or':  #### underestimate max
            self.max_min_index[index_, 0] = 3
            if nonlinear_flag:
                g = self.e_max(list_function, 1e-100)
            else:
                g = self.e_max(list_function)
        elif spec.combination_type == 'and':  ### underestimate min
            self.max_min_index[index_, 0] = 2
            g = self.log_min(list_function)
        self.f_a[index_, 0] = g
        index_ = index_ + 1
        list_index = sorted(list_index)
        self.var_index.append(list_index)
        return index_

    def calculate_x_sub(self, spec, index, state, X_sub, X_robust):
        index_ = index
        #### deal with 'always' and 'eventually'
        if spec.timesteps == list(range(self.K + 1)):
            if isinstance(spec.subformula_list[0], NonlinearPredicate):
                tmp = np.zeros(self.K)
                for k in range(self.K):
                    tmp[k] = spec.subformula_list[0].g(state[:, k])
            elif isinstance(spec.subformula_list[0], LinearPredicate):
                tmp = spec.subformula_list[0].a.T @ state - spec.subformula_list[0].b
            else:
                index_ = self.calculate_x_sub(spec.subformula_list[0], index_, state, X_sub, X_robust)
                tmp = X_sub[index_ - 1, :]
            if spec.combination_type == "and":
                X_sub[index_, 0] = tmp[0]
                for k in range(1, self.K):
                    X_sub[index_, k] = np.min(
                        np.array([tmp[k], X_sub[index_, k - 1]]))
            elif spec.combination_type == "or":
                X_sub[index_, 0] = tmp[0]
                for k in range(1, self.K):
                    X_sub[index_, k] = np.max(
                        np.array([tmp[k], X_sub[index_, k - 1]]))
            return index_ + 1

        list_function = []
        for i, subformula in enumerate(spec.subformula_list):
            if isinstance(subformula, NonlinearPredicate):
                list_function.append(subformula.g(state))
            elif isinstance(subformula, LinearPredicate):
                list_function.append(subformula.a.T @ state - subformula.b)
            else:
                index_ = self.calculate_x_sub(subformula, index_, state, X_sub, X_robust)
                list_function.append(X_sub[index_ - 1, :])
        if index_ < self.n_x_sub:
            for k in range(self.K):
                np_list_function = np.zeros(len(list_function))
                for i in range(len(list_function)):
                    np_list_function[i] = list_function[i][k]
                if spec.combination_type == 'or':  #### underestimate max
                    X_sub[index_, k] = self.e_max(np_list_function)
                elif spec.combination_type == 'and':  ### underestimate min
                    X_sub[index_, k] = self.log_min(np_list_function)
        else:
            np_list_function = np.zeros(len(list_function))
            for i in range(len(list_function)):
                np_list_function[i] = list_function[i][-1]
            if spec.combination_type == 'or':  #### underestimate max
                X_robust[index_-self.n_x_sub, 0] = self.e_max(np_list_function)
            elif spec.combination_type == 'and':  ### underestimate min
                X_robust[index_-self.n_x_sub, 0] = self.log_min(np_list_function)
        index_ = index_ + 1
        return index_

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

    def initialize_trajectory(self, X, X_sub, X_robust, U):
        """
        Initialize the trajectory.

        :param X: Numpy array of states to be initialized
        :param U: Numpy array of inputs to be initialized
        :return: The initialized X and U
        """
        np.random.seed(seed=5)
        for k in range(self.K):
            alpha1 = (self.K - 1 - k) / (self.K - 1)
            alpha2 = k / (self.K - 1)
            X[0:2, k] = alpha1 * self.x_init[0:2] + alpha2 * self.x_final[0:2]  # + np.random.random(2)/100
            X[2:4, k] = alpha1 * self.x_init[2:4] + alpha2 * self.x_final[2:4]

        # X = np.load("x2.npy")
        # X[2:4, :] = 0
        # U = np.load("u.npy")
        # X = np.zeros((self.n_x, self.K))
        self.calculate_x_sub(self.spec, 0, X, X_sub, X_robust)
        self.x_sub_init = X_sub[:, 0]

        U = np.zeros((self.n_u, self.K))
        return X, X_sub, X_robust, U

    def get_objective(self, X_v, X_robust_v, U_v, X_last_p, X_robust_last_p, U_last_p):
        """
        Get model specific objective to be minimized.

        :param X_v: cvx variable for current states
        :param U_v: cvx variable for current inputs
        :param X_last_p: cvx parameter for last states
        :param U_last_p: cvx parameter for last inputs
        :return: A cvx objective function.
        """
        return cvx.Minimize(1e-5 * cvx.sum_squares(U_v) - 1e-4 * X_robust_v[-1, 0])

    def get_constraints(self, X_v, X_robust_v, U_v, X_last_p, X_robust_last_p, U_last_p):
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
            X_v[:, 0] == self.x_init
            # X_v[:, self.K - 1] == self.x_final
        ]

        constraints += [
            # State constraints:
            X_v[0, :] <= self.x_max[0], X_v[0, :] >= self.x_min[0],
            X_v[1, :] <= self.x_max[1], X_v[1, :] >= self.x_min[1],
            X_v[2, :] <= self.x_max[2], X_v[2, :] >= self.x_min[2],
            X_v[3, :] <= self.x_max[3], X_v[3, :] >= self.x_min[3],
            X_robust_v[0, 0] >= 0,
            U_v[0, :] <= self.u_max[0], U_v[0, :] >= self.u_min[0],
            U_v[1, :] <= self.u_max[1], U_v[1, :] >= self.u_min[1]
        ]

        return constraints

    def add_sub_dynamics_constrains(self, X_v, X_sub_v, X_robust_v, U_v, X_last_p,
                                    X_sub_last_p, X_robust_last_p, U_last_p, nu_sub, nu_robust,
                                    par_sub_dynamics_matrix):
        constraints = []
        C = self.C
        f_a = self.f_a
        f_b = self.f_b
        var = self.var
        var_index = self.var_index
        par_sub_dynamics_matrix.append(cvx.Parameter(shape=self.x_sub_init.shape, value=self.x_sub_init))
        constraints += [X_sub_v[:, 0] == par_sub_dynamics_matrix[-1] + nu_sub[:, 0]]

        for k in range(1, self.K):
            input_value = np.hstack((X_last_p.value[:, k], X_sub_last_p.value[:, k],
                                     X_sub_last_p.value[:, k - 1], U_last_p.value[:, k]))
            input_value = input_value.reshape(-1)

            input_value_par = cvx.hstack((X_last_p[:, k], X_sub_last_p[:, k],
                                          X_sub_last_p[:, k - 1], U_last_p[:, k]))

            input_value_par = cvx.reshape(input_value_par, (self.n_x + 2 * self.n_x_sub + self.n_u, 1))

            input_var = cvx.hstack((X_v[:, k], X_sub_v[:, k], X_sub_v[:, k - 1], U_v[:, k]))
            input_var = cvx.reshape(input_var, (self.n_x + 2 * self.n_x_sub + self.n_u, 1))

            for i in range(self.n_x_sub):
                if self.max_min_index[i, 0] == 0:
                    f_, f_gradient = self.smin(f_a[i, 0], f_b[i, 0], C, input_value[var_index[i]], var[var_index[i], 0])
                elif self.max_min_index[i, 0] == 1:
                    f_, f_gradient = self.smin(-f_a[i, 0], -f_b[i, 0], C, input_value[var_index[i]],
                                               var[var_index[i], 0])
                    f_ = -f_
                    f_gradient = -f_gradient
                else:  ###if self.max_min_index[i, 0] == 2 or self.max_min_index[i, 0] == 3:
                    f_ = sp.zeros(1, 1)
                    f_[0, 0] = f_a[i, 0]
                    if f_[0, 0] not in self.known_gradient:
                        f_gradient = f_.jacobian(var[var_index[i], 0])
                        self.known_gradient[f_[0, 0]] = f_gradient
                    else:
                        f_gradient = self.known_gradient[f_[0, 0]]
                # f_func = sp.lambdify((var[var_index[i], 0],), f_, 'numpy')
                # f_gradient_func = sp.lambdify((var[var_index[i], 0],), f_gradient, 'numpy')
                if (str(var[var_index[i], 0]), f_[0, 0]) not in self.known_lambda:
                    f_func = sp.lambdify((var[var_index[i], 0],), f_, 'numpy')
                    self.known_lambda[(str(var[var_index[i], 0]), f_[0, 0])] = f_func
                else:
                    f_func = self.known_lambda[(str(var[var_index[i], 0]), f_[0, 0])]

                if (str(var[var_index[i], 0]), f_gradient[0, 0]) not in self.known_lambda:
                    f_gradient_func = sp.lambdify((var[var_index[i], 0],), f_gradient, 'numpy')
                    self.known_lambda[(str(var[var_index[i], 0]), f_gradient[0, 0])] = f_gradient_func
                else:
                    f_gradient_func = self.known_lambda[(str(var[var_index[i], 0]), f_gradient[0, 0])]
                M1 = f_func(input_value[var_index[i]])
                if M1.shape == (1, 1):
                    M1 = M1[0, 0]
                par_sub_dynamics_matrix.append(cvx.Parameter(shape=(),
                                                             value=M1))
                tt_ = f_gradient_func(input_value[var_index[i]])
                par_sub_dynamics_matrix.append(cvx.Parameter(shape=f_gradient_func(input_value[var_index[i]]).shape,
                                                             value=f_gradient_func(input_value[var_index[i]])))
                # if f_gradient.is_constant():
                constraints += [
                    X_sub_v[i, k] == par_sub_dynamics_matrix[-2] + par_sub_dynamics_matrix[-1] @ (
                            input_var[var_index[i]] - input_value_par[var_index[i]]) + nu_sub[i, k]]
                # else:
                #     constraints += [
                #         X_sub_v[i, k] == par_sub_dynamics_matrix[-2] + par_sub_dynamics_matrix[-1] @ (
                #                 input_var[var_index[i]] - input_value_par[var_index[i]])]
        # constraints += [
        #     X_robust_v[0, 0] == X_sub_v[-1, -1]]

        ################ robust calculation
        input_value = np.hstack((X_last_p.value[:, -1], X_sub_last_p.value[:, -1]))
        input_value = input_value.reshape(-1)
        input_value_par = cvx.hstack((X_last_p[:, -1], X_sub_last_p[:, -1]))
        input_value_par = cvx.reshape(input_value_par, (self.n_x + self.n_x_sub, 1))
        input_var = cvx.hstack((X_v[:, -1], X_sub_v[:, -1]))
        input_var = cvx.reshape(input_var, (self.n_x + self.n_x_sub, 1))
        for i in range(self.n_x_sub, self.n_x_sub + self.n_x_robust):
            f_ = sp.zeros(1, 1)
            f_[0, 0] = f_a[i, 0]
            if f_[0, 0] not in self.known_gradient:
                f_gradient = f_.jacobian(var[var_index[i], 0])
                self.known_gradient[f_[0, 0]] = f_gradient
            else:
                f_gradient = self.known_gradient[f_[0, 0]]
            if (str(var[var_index[i], 0]), f_[0, 0]) not in self.known_lambda:
                f_func = sp.lambdify((var[var_index[i], 0],), f_, 'numpy')
                self.known_lambda[(str(var[var_index[i], 0]), f_[0, 0])] = f_func
            else:
                f_func = self.known_lambda[(str(var[var_index[i], 0]), f_[0, 0])]

            if (str(var[var_index[i], 0]), f_gradient[0, 0]) not in self.known_lambda:
                f_gradient_func = sp.lambdify((var[var_index[i], 0],), f_gradient, 'numpy')
                self.known_lambda[(str(var[var_index[i], 0]), f_gradient[0, 0])] = f_gradient_func
            else:
                f_gradient_func = self.known_lambda[(str(var[var_index[i], 0]), f_gradient[0, 0])]
            M1 = f_func(input_value[var_index[i]])
            if M1.shape == (1, 1):
                M1 = M1[0, 0]
            par_sub_dynamics_matrix.append(cvx.Parameter(shape=(),
                                                         value=M1))

            par_sub_dynamics_matrix.append(cvx.Parameter(shape=f_gradient_func(input_value[var_index[i]]).shape,
                                                         value=f_gradient_func(input_value[var_index[i]])))
            constraints += [
                X_robust_v[i - self.n_x_sub, 0] == par_sub_dynamics_matrix[-2] + par_sub_dynamics_matrix[-1] @ (
                        input_var[var_index[i]] - input_value_par[var_index[i]]) + nu_robust[i - self.n_x_sub, 0]]
        return constraints

    def get_sub_dynamics_constrains(self, X_last_p,
                                    X_sub_last_p, X_robust_last_p, U_last_p, par_sub_dynamics_matrix):
        constraints = []
        C = self.C
        f_a = self.f_a
        f_b = self.f_b
        var = self.var
        var_index = self.var_index
        j = 0
        par_sub_dynamics_matrix[j].value = self.x_sub_init
        j += 1

        for k in range(1, self.K):
            input_value = np.hstack((X_last_p.value[:, k], X_sub_last_p.value[:, k],
                                     X_sub_last_p.value[:, k - 1], U_last_p.value[:, k]))
            input_value = input_value.reshape(-1)
            input_value_par = cvx.hstack((X_last_p[:, k], X_sub_last_p[:, k],
                                          X_sub_last_p[:, k - 1], U_last_p[:, k]))

            for i in range(self.n_x_sub):
                if self.max_min_index[i, 0] == 0:
                    f_, f_gradient = self.smin(f_a[i, 0], f_b[i, 0], C, input_value[var_index[i]], var[var_index[i], 0])
                elif self.max_min_index[i, 0] == 1:
                    f_, f_gradient = self.smin(-f_a[i, 0], -f_b[i, 0], C, input_value[var_index[i]],
                                               var[var_index[i], 0])
                    f_ = -f_
                    f_gradient = -f_gradient
                else:  ###if self.max_min_index[i, 0] == 2 or self.max_min_index[i, 0] == 3:
                    f_ = sp.zeros(1, 1)
                    f_[0, 0] = f_a[i, 0]
                    if f_[0, 0] not in self.known_gradient:
                        f_gradient = f_.jacobian(var[var_index[i], 0])
                        self.known_gradient[f_[0, 0]] = f_gradient
                    else:
                        f_gradient = self.known_gradient[f_[0, 0]]
                # f_func = sp.lambdify((var[var_index[i], 0],), f_, 'numpy')
                # f_gradient_func = sp.lambdify((var[var_index[i], 0],), f_gradient, 'numpy')
                if (str(var[var_index[i], 0]), f_[0, 0]) not in self.known_lambda:
                    f_func = sp.lambdify((var[var_index[i], 0],), f_, 'numpy')
                    self.known_lambda[(str(var[var_index[i], 0]), f_[0, 0])] = f_func
                else:
                    f_func = self.known_lambda[(str(var[var_index[i], 0]), f_[0, 0])]

                if (str(var[var_index[i], 0]), f_gradient[0, 0]) not in self.known_lambda:
                    f_gradient_func = sp.lambdify((var[var_index[i], 0],), f_gradient, 'numpy')
                    self.known_lambda[(str(var[var_index[i], 0]), f_gradient[0, 0])] = f_gradient_func
                else:
                    f_gradient_func = self.known_lambda[(str(var[var_index[i], 0]), f_gradient[0, 0])]

                M1 = f_func(input_value[var_index[i]])  #####TODO
                if M1.shape == (1, 1):
                    M1 = M1[0, 0]
                par_sub_dynamics_matrix[j].value = M1
                j += 1
                par_sub_dynamics_matrix[j].value = f_gradient_func(input_value[var_index[i]])
                j += 1

        ################ robust calculation
        input_value = np.hstack((X_last_p.value[:, -1], X_sub_last_p.value[:, -1]))
        input_value = input_value.reshape(-1)
        input_value_par = cvx.hstack((X_last_p[:, k], X_sub_last_p[:, k]))
        input_value_par = cvx.reshape(input_value_par, (self.n_x + self.n_x_sub, 1))
        for i in range(self.n_x_sub, self.n_x_sub + self.n_x_robust):
            f_ = sp.zeros(1, 1)
            f_[0, 0] = f_a[i, 0]
            if f_[0, 0] not in self.known_gradient:
                f_gradient = f_.jacobian(var[var_index[i], 0])
                self.known_gradient[f_[0, 0]] = f_gradient
            else:
                f_gradient = self.known_gradient[f_[0, 0]]
            if (str(var[var_index[i], 0]), f_[0, 0]) not in self.known_lambda:
                f_func = sp.lambdify((var[var_index[i], 0],), f_, 'numpy')
                self.known_lambda[(str(var[var_index[i], 0]), f_[0, 0])] = f_func
            else:
                f_func = self.known_lambda[(str(var[var_index[i], 0]), f_[0, 0])]

            if (str(var[var_index[i], 0]), f_gradient[0, 0]) not in self.known_lambda:
                f_gradient_func = sp.lambdify((var[var_index[i], 0],), f_gradient, 'numpy')
                self.known_lambda[(str(var[var_index[i], 0]), f_gradient[0, 0])] = f_gradient_func
            else:
                f_gradient_func = self.known_lambda[(str(var[var_index[i], 0]), f_gradient[0, 0])]
            M1 = f_func(input_value[var_index[i]])
            if M1.shape == (1, 1):
                M1 = M1[0, 0]
            par_sub_dynamics_matrix[j].value = M1
            j += 1
            par_sub_dynamics_matrix[j].value = f_gradient_func(input_value[var_index[i]])
            j += 1

        return constraints

    def get_linear_cost(self):
        return 0

    def get_nonlinear_cost(self, X=None, U=None):
        return 0

    def calculate_subdynamics(self, X_nl):
        X_sub_nl = np.empty(shape=[self.n_x_sub, self.K])
        X_robust_nl = np.empty(shape=[self.n_x_robust, 1])
        self.calculate_x_sub(self.spec, 0, X_nl, X_sub_nl, X_robust_nl)
        self.x_sub_init = X_sub_nl[:, 0]

        return X_sub_nl, X_robust_nl

    def smin(self, a_func, b_func, C, input_value, input_variable):
        #### input_value: x_1, x_2, v_1, v_2, x_sub_1, x_sub_2, x_sub_3, x_sub_4, x_sub_prev_1, x_sub_prev_2, x_sub_prev_3, x_sub_prev_4,
        #### u_1, u_2
        value_func = sp.zeros(1, 1)
        gradient_func = None

        if (str(input_variable), a_func) not in self.known_lambda:
            a_lamdba = sp.lambdify((input_variable,), a_func, 'numpy')
            self.known_lambda[(str(input_variable), a_func)] = a_lamdba
        else:
            a_lamdba = self.known_lambda[(str(input_variable), a_func)]

        if (str(input_variable), b_func) not in self.known_lambda:
            b_lamdba = sp.lambdify((input_variable,), b_func, 'numpy')
            self.known_lambda[(str(input_variable), b_func)] = b_lamdba
        else:
            b_lamdba = self.known_lambda[(str(input_variable), b_func)]

        if a_lamdba(input_value) - b_lamdba(input_value) <= -C:
            value_func[0, 0] = a_func
        elif a_lamdba(input_value) - b_lamdba(input_value) >= C:
            value_func[0, 0] = b_func
        else:
            h = sp.zeros(1, 1)
            g_func = sp.zeros(1, 1)
            h[0, 0] = 1 / 2 + (a_func - b_func) / 2 / C
            g_func[0, 0] = a_func * (1 - h[0, 0]) + h[0, 0] * b_func - C * h[0, 0] * (1 - h[0, 0])
            value_func[0, 0] = g_func

        if value_func[0, 0] not in self.known_gradient:
            gradient_func = value_func.jacobian(input_variable)
            self.known_gradient[value_func[0, 0]] = gradient_func
        else:
            gradient_func = self.known_gradient[value_func[0, 0]]

        return value_func, gradient_func

    def add_to_plot(self, ax):
        # Add red rectangles for the obstacles
        for obstacle in self.obstacles:
            ax.add_patch(make_circle_patch(obstacle, 1.0, color='k', alpha=0.5, zorder=-1))

        # Use the color cycle to choose the colors of each target group
        # (note that this won't work for more than 10 target groups)
        colors = plt.cm.tab10.colors
        for i, target_group in enumerate(self.targets):
            color = colors[i]
            for target in target_group:
                ax.add_patch(make_circle_patch(target, 1.0, color=color, alpha=0.7, zorder=-1))
        color = colors[i + 1]
        ax.add_patch(make_circle_patch(self.goal, 1.0, color=color, alpha=0.7, zorder=-1))
        # set the field of view
        ax.set_xlim((0, 10))
        ax.set_ylim((0, 10))
        ax.set_aspect('equal')
