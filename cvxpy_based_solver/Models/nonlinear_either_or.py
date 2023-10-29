from .base_model import Model
from .base_model import (MovingNonlinearPredicate, inside_moving_circle_formula)

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
    n_moving_predicates = 0

    def __init__(self, K, t_f_guess, center_var_number=2):
        """
        A large r_scale for a small scale problem will
        ead to numerical problems as parameters become excessively small
        and (it seems) precision is lost in the dynamics.
        """
        Model.__init__(self, K, t_f_guess, max_k=15, smin_C=0.1)
        self.center_var_number = center_var_number

        self.x_init = np.array([2.0, 2.0, 0, 0])
        self.x_final = np.array([7.5, 8.5, 0, 0])

        self.u_min = np.array([-10, -10])
        self.u_max = np.array([10, 10])
        self.x_min = np.array([-5.0, -5.0, -5.0, -5.0])
        self.x_max = np.array([10.0, 10.0, 5.0, 5.0])

        goal = np.array([[7.5, 8.5], [1.5, 6.5], [7.5, 5.0]])  # goal center and radius
        goal_rad = np.array([0.5, 0.5, 0.5])
        obs = np.array([[5.5, 5.0], [8.0, 6.5]])  # obstacle center and radius
        obs_rad = np.array([1.5, 0.5])

        self.goal_center = goal
        self.goal_radius = goal_rad

        self.obstacle_center = obs
        self.obstacle_radius = obs_rad

        at_goal = inside_circle_formula(self.goal_center[0, :], self.goal_radius[0], 0, 1, 4)
        at_goal_1 = inside_circle_formula(self.goal_center[1, :], self.goal_radius[1], 0, 1, 4)
        at_goal_2 = inside_circle_formula(self.goal_center[2, :], self.goal_radius[2], 0, 1, 4)
        either_goal = at_goal_1 | at_goal_2
        either_goal.simplify()

        # Obstacle Avoidance
        at_obstacle_1 = inside_circle_formula(self.obstacle_center[0, :],
                                              self.obstacle_radius[0], 0, 1, 4)
        not_at_obstacle_1 = at_obstacle_1.negation()
        at_obstacle_2 = inside_circle_formula(self.obstacle_center[1, :],
                                              self.obstacle_radius[1], 0, 1, 4)
        not_at_obstacle_2 = at_obstacle_2.negation()

        # Put all of the constraints together in one specification
        self.spec = not_at_obstacle_1.always(0, self.K) & not_at_obstacle_2.always(0, self.K) & \
                    (either_goal).eventually(0, self.K) & at_goal.eventually(0, self.K)
        self.flatten_before_sub(self.spec)

        self.robustness_index = []
        self.calculate_n_x_sub(self.spec, flag=0)
        self.zero_robustness = 0
        if self.n_x_robust == 0:
            self.n_x_robust = 1
            self.zero_robustness = 1

        self.x_sub_init = np.zeros((self.n_x_sub, 1))
        self.x_robust_init = np.zeros(self.n_x_robust)

        self.obstacle_moving_position = np.zeros((self.n_moving_predicates, self.center_var_number, K))
        self.index_moving_predicates = 0

        str_variable = ''

        ### no until (now)

        for i in range(self.n_x + 2 * self.n_x_sub +
                       self.n_x_robust + self.center_var_number*self.n_moving_predicates):  ### x, x_sub, x_sub_previous, x_robust, position_predicate
            str_variable += 'x_' + str(i)
            if i != self.n_x + 2 * self.n_x_sub +\
                    self.n_x_robust + self.center_var_number*self.n_moving_predicates - 1:
                str_variable += ', '

        self.var = sp.Matrix(sp.symbols(str_variable, real=True))
        self.f_a = sp.zeros(self.n_x_sub + self.n_x_robust, 1)
        self.f_b = sp.zeros(self.n_x_sub + self.n_x_robust, 1)

        self.var_index = []
        self.var_lambda_index = []
        self.max_min_index = np.zeros((self.n_x_sub + self.n_x_robust, 1))
        self.moving_obstacle_index_list = np.zeros((self.n_x_sub + self.n_x_robust, 1))
        self.add_interpolate_function(self.spec, 0, self.var)

    def calculate_n_x_sub(self, spec, flag): ## flag = 0 before "always", flag = 1 after "always"
        if isinstance(spec, MovingNonlinearPredicate):
            self.n_moving_predicates += 1
            return
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
            if isinstance(spec.subformula_list[0], MovingNonlinearPredicate):
                self.obstacle_moving_position[self.index_moving_predicates, :, :] =\
                    spec.subformula_list[0].center_position
                tmp_position_index = self.n_x + 2 * self.n_x_sub + self.n_x_robust + \
                                     self.center_var_number * self.index_moving_predicates - 1
                self.var_index.append([0, 1, self.n_x + self.n_x_sub + index_])
                list_ = [0, 1, self.n_x + self.n_x_sub + index_] + list(range(tmp_position_index, tmp_position_index+self.center_var_number))
                self.var_lambda_index.append(list_)
                tt = var[0:4] + var[tmp_position_index:tmp_position_index+self.center_var_number]
                self.f_a[index_, 0] = spec.subformula_list[0].g(tt)
                self.f_b[index_, 0] = var[self.n_x + self.n_x_sub + index_, 0]
                self.moving_obstacle_index_list[index_, 0] = 1
                self.index_moving_predicates += 1
            elif isinstance(spec.subformula_list[0], NonlinearPredicate):
                self.var_index.append([0, 1, self.n_x + self.n_x_sub + index_])
                self.var_lambda_index.append([0, 1, self.n_x + self.n_x_sub + index_])
                self.f_a[index_, 0] = spec.subformula_list[0].g(var[0:4])
                self.f_b[index_, 0] = var[self.n_x + self.n_x_sub + index_, 0]
                ### TODO now 0,1 are manually setting, later use STL to get it.
            elif isinstance(spec.subformula_list[0], LinearPredicate):
                self.var_index.append([0, 1, self.n_x + self.n_x_sub + index_])
                self.var_lambda_index.append([0, 1, self.n_x + self.n_x_sub + index_])
                self.f_a[index_, 0] = spec.subformula_list[0].a.T @ var[0:4] - spec.subformula_list[0].b
                self.f_b[index_, 0] = var[self.n_x + self.n_x_sub + index_, 0]
            else:
                index_ = self.add_interpolate_function(spec.subformula_list[0], index_, var)
                self.f_a[index_, 0] = var[self.n_x + self.n_x_sub + index_, 0] ### previous self
                self.f_b[index_, 0] = var[self.n_x + index_ - 1, 0] ### current last sub state
                self.var_index.append([self.n_x + self.n_x_sub + index_, self.n_x + index_ - 1])
                self.var_lambda_index.append([self.n_x + self.n_x_sub + index_, self.n_x + index_ - 1])
            if spec.combination_type == 'or':  #### smax
                self.max_min_index[index_, 0] = 1
            if self.zero_robustness:
                if index_ == self.n_x_sub - 1:
                    self.f_a[index_ + 1, 0] = var[self.n_x + index_, 0]
                    self.max_min_index[index_ + 1, 0] = 2
                    self.var_index.append([self.n_x + index_])
                    self.var_lambda_index.append([self.n_x + index_])
            return index_ + 1

        list_function = []
        list_index = []
        list_lambda_index = []
        for i, subformula in enumerate(spec.subformula_list):
            if isinstance(subformula, MovingNonlinearPredicate):
                self.obstacle_moving_position[self.index_moving_predicates, :, :] = subformula.center_position
                tmp_position_index = self.n_x + 2 * self.n_x_sub + self.n_x_robust + \
                                     self.center_var_number * self.index_moving_predicates - 1

                tt = var[0:4] + var[tmp_position_index:tmp_position_index+self.center_var_number]
                list_function.append(subformula.g(tt))
                if not 0 in list_index:
                    list_index.append(0)
                    list_lambda_index.append(0)
                if not 1 in list_index:
                    list_index.append(1)
                    list_lambda_index.append(1)
                if not tmp_position_index in list_lambda_index:
                    list_lambda_index += list(range(tmp_position_index, tmp_position_index+self.center_var_number))
                self.moving_obstacle_index_list[index_, 0] = 1
                self.index_moving_predicates += 1
            elif isinstance(subformula, NonlinearPredicate):
                list_function.append(subformula.g(var[0:4]))
                if not 0 in list_index:
                    list_index.append(0)
                    list_lambda_index.append(0)
                if not 1 in list_index:
                    list_index.append(1)
                    list_lambda_index.append(1)
            elif isinstance(subformula, LinearPredicate):
                tmp = subformula.a.T @ var[0:4] - subformula.b
                list_function.append(tmp[0])
                if not 0 in list_index:
                    list_index.append(0)
                    list_lambda_index.append(0)
                if not 1 in list_index:
                    list_index.append(1)
                    list_lambda_index.append(1)
            else:
                index_ = self.add_interpolate_function(subformula, index_, var)
                list_function.append(var[self.n_x + index_ - 1, 0])
                list_index.append(self.n_x + index_ - 1)
                list_lambda_index.append(self.n_x + index_ - 1)
        g = 0
        if spec.combination_type == 'or':  #### underestimate max
            self.max_min_index[index_, 0] = 3
            g = self.e_max(list_function)
        elif spec.combination_type == 'and':  ### underestimate min
            self.max_min_index[index_, 0] = 2
            g = self.log_min(list_function)
        self.f_a[index_, 0] = g
        index_ = index_ + 1
        list_index = sorted(list_index)
        self.var_index.append(list_index)
        self.var_lambda_index.append(list_lambda_index)
        return index_

    def calculate_x_sub(self, spec, index, state, X_sub, X_robust):
        index_ = index
        #### deal with 'always' and 'eventually'
        if spec.timesteps == list(range(self.K + 1)):
            if isinstance(spec.subformula_list[0], MovingNonlinearPredicate):
                tmp = np.zeros(self.K)
                for k in range(self.K):
                    tmp[k] = spec.subformula_list[0].g(np.hstack((state[:, k], spec.subformula_list[0].center_position[:, k])))
            elif isinstance(spec.subformula_list[0], NonlinearPredicate):
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
            if self.zero_robustness:
                if index_ == self.n_x_sub - 1:
                    X_robust[index_ - self.n_x_sub, 0] = X_sub[index_ - 1, -1]
            return index_ + 1

        list_function = []
        for i, subformula in enumerate(spec.subformula_list):
            if isinstance(subformula, MovingNonlinearPredicate):
                tmp = np.zeros((1, self.K))
                for k in range(self.K):
                    tmp[0, k] = subformula.g(np.hstack((state[:, k], subformula.center_position[:, k])))
                list_function.append(tmp)
            elif isinstance(subformula, NonlinearPredicate):
                tmp2 = subformula.g(state)
                tmp = np.zeros((1, self.K))
                tmp[0, :] = tmp2[:]
                list_function.append(tmp)
            elif isinstance(subformula, LinearPredicate):
                list_function.append(subformula.a.T @ state - subformula.b)
            else:
                index_ = self.calculate_x_sub(subformula, index_, state, X_sub, X_robust)
                tmp = np.zeros((1, self.K))
                tmp[0, :] = X_sub[index_ - 1, :]
                list_function.append(tmp)
        if index_ < self.n_x_sub:
            for k in range(self.K):
                np_list_function = np.zeros(len(list_function))
                for i in range(len(list_function)):
                    np_list_function[i] = list_function[i][0, k]
                if spec.combination_type == 'or':  #### underestimate max
                    X_sub[index_, k] = self.e_max(np_list_function)
                elif spec.combination_type == 'and':  ### underestimate min
                    X_sub[index_, k] = self.log_min(np_list_function)
        else:
            np_list_function = np.zeros(len(list_function))
            for i in range(len(list_function)):
                np_list_function[i] = list_function[i][0, -1]
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
        #np.random.seed(seed=5)
        for k in range(self.K):
            alpha1 = (self.K - 1 - k) / (self.K - 1)
            alpha2 = k / (self.K - 1)
            X[0:2, k] = alpha1 * self.x_init[0:2] + alpha2 * self.x_final[0:2]  # + np.random.random(2)/100
            X[2:4, k] = alpha1 * self.x_init[2:4] + alpha2 * self.x_final[2:4]

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
        return cvx.Minimize(1e-3 * cvx.sum_squares(U_v) - X_robust_v[-1, 0])

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
        self.index_moving_predicates = 0
        C = self.C
        f_a = self.f_a
        f_b = self.f_b
        var = self.var
        var_index = self.var_index
        var_lambda_index = self.var_lambda_index
        par_sub_dynamics_matrix.append(cvx.Parameter(shape=self.x_sub_init.shape, value=self.x_sub_init))
        constraints += [X_sub_v[:, 0] == par_sub_dynamics_matrix[-1] + nu_sub[:, 0]]

        for k in range(1, self.K):
            input_value = np.hstack((X_last_p.value[:, k], X_sub_last_p.value[:, k],
                                     X_sub_last_p.value[:, k - 1],
                                     self.obstacle_moving_position[:, :, k].reshape(-1)))
            input_value = input_value.reshape(-1)

            input_value_par = cvx.hstack((X_last_p[:, k], X_sub_last_p[:, k],
                                          X_sub_last_p[:, k - 1], U_last_p[:, k]))

            input_value_par = cvx.reshape(input_value_par, (self.n_x + 2 * self.n_x_sub + self.n_u, 1))

            input_var = cvx.hstack((X_v[:, k], X_sub_v[:, k], X_sub_v[:, k - 1], U_v[:, k]))
            input_var = cvx.reshape(input_var, (self.n_x + 2 * self.n_x_sub + self.n_u, 1))
            self.index_moving_predicates = 0
            for i in range(self.n_x_sub):
                M1, M2 = self.get_sub_dynamics_constrains_parameters(input_value, var, var_index, var_lambda_index, i)
                par_sub_dynamics_matrix.append(cvx.Parameter(shape=(), value=M1))
                par_sub_dynamics_matrix.append(cvx.Parameter(shape=M2.shape, value=M2))
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
        input_value = np.hstack((X_last_p.value[:, -1], X_sub_last_p.value[:, -1], X_robust_last_p.value[:, 0]))
        input_value = input_value.reshape(-1)
        input_value_par = cvx.hstack((X_last_p[:, -1], X_sub_last_p[:, -1], X_robust_last_p[:, 0]))
        input_value_par = cvx.reshape(input_value_par, (self.n_x + self.n_x_sub + self.n_x_robust, 1))
        input_var = cvx.hstack((X_v[:, -1], X_sub_v[:, -1], X_robust_v[:, 0]))
        input_var = cvx.reshape(input_var, (self.n_x + self.n_x_sub + self.n_x_robust, 1))
        for i in range(self.n_x_sub, self.n_x_sub + self.n_x_robust):
            f_ = sp.zeros(1, 1)
            f_[0, 0] = f_a[i, 0]
            f_gradient = self.cal_f_gradient(var[var_index[i], 0], f_)
            f_func = self.f_lambdify((var[var_index[i], 0],), f_)
            f_gradient_func = self.f_lambdify((var[var_index[i], 0],), f_gradient)
            M1 = f_func(input_value[var_index[i]])
            if M1.shape == (1, 1):
                M1 = M1[0, 0]
            par_sub_dynamics_matrix.append(cvx.Parameter(shape=(),
                                                         value=M1))

            M2 = f_gradient_func(input_value[var_index[i]])
            par_sub_dynamics_matrix.append(cvx.Parameter(shape=M2.shape,
                                                         value=M2))
            constraints += [
                X_robust_v[i - self.n_x_sub, 0] == par_sub_dynamics_matrix[-2] + par_sub_dynamics_matrix[-1] @ (
                        input_var[var_index[i]] - input_value_par[var_index[i]]) + nu_robust[i - self.n_x_sub, 0]]
        return constraints

    def get_sub_dynamics_constrains(self, X_last_p,
                                    X_sub_last_p, X_robust_last_p, U_last_p, par_sub_dynamics_matrix):
        constraints = []
        self.index_moving_predicates = 0
        C = self.C
        f_a = self.f_a
        f_b = self.f_b
        var = self.var
        var_index = self.var_index
        var_lambda_index = self.var_lambda_index
        j = 0
        par_sub_dynamics_matrix[j].value = self.x_sub_init
        j += 1

        for k in range(1, self.K):
            input_value = np.hstack((X_last_p.value[:, k], X_sub_last_p.value[:, k],
                                     X_sub_last_p.value[:, k - 1],
                                     self.obstacle_moving_position[:, :, k].reshape(-1)))
            input_value = input_value.reshape(-1)
            self.index_moving_predicates = 0
            for i in range(self.n_x_sub):
                M1, M2 = self.get_sub_dynamics_constrains_parameters(input_value, var, var_index, var_lambda_index, i)
                par_sub_dynamics_matrix[j].value = M1
                j += 1
                par_sub_dynamics_matrix[j].value = M2
                j += 1

        ################ robust calculation

        input_value = np.hstack((X_last_p.value[:, -1], X_sub_last_p.value[:, -1], X_robust_last_p.value[:, 0]))
        input_value = input_value.reshape(-1)
        for i in range(self.n_x_sub, self.n_x_sub + self.n_x_robust):
            f_ = sp.zeros(1, 1)
            f_[0, 0] = f_a[i, 0]
            f_gradient = self.cal_f_gradient(var[var_index[i], 0], f_)
            f_func = self.f_lambdify((var[var_index[i], 0],), f_)
            f_gradient_func = self.f_lambdify((var[var_index[i], 0],), f_gradient)
            M1 = f_func(input_value[var_index[i]])
            if M1.shape == (1, 1):
                M1 = M1[0, 0]
            M2 = f_gradient_func(input_value[var_index[i]])
            par_sub_dynamics_matrix[j].value = M1
            j += 1
            par_sub_dynamics_matrix[j].value = M2
            j += 1

        return constraints

    def get_sub_dynamics_constrains_parameters(self, input_value, var, var_index, var_lambda_index, i):

        if self.max_min_index[i, 0] == 0:
            f_, f_gradient = self.smin(self.f_a[i, 0], self.f_b[i, 0], self.C,
                                       input_value[var_lambda_index[i]], var[var_lambda_index[i], 0],
                                       var[var_index[i], 0])
        elif self.max_min_index[i, 0] == 1:
            f_, f_gradient = self.smin(-self.f_a[i, 0], -self.f_b[i, 0], self.C, input_value[var_lambda_index[i]],
                                       var[var_lambda_index[i], 0],
                                       var[var_index[i], 0])
            f_ = -f_
            f_gradient = -f_gradient
        else:  ###if self.max_min_index[i, 0] == 2 or self.max_min_index[i, 0] == 3:
            f_ = sp.zeros(1, 1)
            f_[0, 0] = self.f_a[i, 0]
            f_gradient = self.cal_f_gradient(var[var_index[i], 0], f_)
        f_func = self.f_lambdify((var[var_lambda_index[i], 0],), f_)
        f_gradient_func = self.f_lambdify((var[var_lambda_index[i], 0],), f_gradient)
        M1 = f_func(input_value[var_lambda_index[i]])
        if M1.shape == (1, 1):
            M1 = M1[0, 0]
        M2 = f_gradient_func(input_value[var_lambda_index[i]])

        return M1, M2

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

    def add_to_plot(self, ax):
        # Make and add circular patches
        obstacle_1 = make_circle_patch(self.obstacle_center[0, :],
                                       self.obstacle_radius[0], color='k', alpha=0.5)
        obstacle_2 = make_circle_patch(self.obstacle_center[1, :],
                                       self.obstacle_radius[1], color='k', alpha=0.5)
        goal = make_circle_patch(self.goal_center[0, :], self.goal_radius[0],
                                 color='green', alpha=0.5)
        goal_1 = make_circle_patch(self.goal_center[1, :], self.goal_radius[1],
                                   color='blue', alpha=0.5)
        goal_2 = make_circle_patch(self.goal_center[2, :], self.goal_radius[2],
                                   color='blue', alpha=0.5)

        ax.add_patch(obstacle_1)
        ax.add_patch(obstacle_2)
        ax.add_patch(goal)
        ax.add_patch(goal_1)
        ax.add_patch(goal_2)

        # set the field of view
        ax.set_xlim((0, 10))
        ax.set_ylim((0, 10))
        ax.set_aspect('equal')
