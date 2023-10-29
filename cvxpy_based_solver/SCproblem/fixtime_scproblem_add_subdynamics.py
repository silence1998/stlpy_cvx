import cvxpy as cvx
from .base_scproblem import SCProblem


class FixTimeSubDynamicSCProblem(SCProblem):
    """
    Defines a standard Successive Convexification problem with fixed time and adds the model specific constraints and objectives.

    :param m: The model object
    :param K: Number of discretization points
    """

    def __init__(self, m, K):
        # Variables:
        self.var = dict()
        self.var['X'] = cvx.Variable((m.n_x, K))
        self.var['X_sub'] = cvx.Variable((m.n_x_sub, K))
        self.var['X_robust'] = cvx.Variable((m.n_x_robust, 1))
        self.var['U'] = cvx.Variable((m.n_u, K))
        self.var['nu'] = cvx.Variable((m.n_x, K - 1))
        self.var['nu_sub'] = cvx.Variable((m.n_x_sub, K))
        self.var['nu_robust'] = cvx.Variable((m.n_x_robust, 1))

        # Parameters:
        self.par = dict()
        self.par['A_bar'] = cvx.Parameter((m.n_x * m.n_x, K - 1))
        self.par['B_bar'] = cvx.Parameter((m.n_x * m.n_u, K - 1))
        self.par['C_bar'] = cvx.Parameter((m.n_x * m.n_u, K - 1))
        self.par['z_bar'] = cvx.Parameter((m.n_x, K - 1))

        self.par['X_last'] = cvx.Parameter((m.n_x, K))
        self.par['X_sub_last'] = cvx.Parameter((m.n_x_sub, K))
        self.par['X_robust_last'] = cvx.Parameter((m.n_x_robust, 1))
        self.par['U_last'] = cvx.Parameter((m.n_u, K))

        self.par_sub_dynamics_matrix = []

        self.par['weight_nu'] = cvx.Parameter(nonneg=True)
        self.par['tr_radius'] = cvx.Parameter(nonneg=True)

        self.m = m
        self.K = K

        self.prob = None

    def init_problem(self):
        # Constraints:
        constraints = []

        # Model:
        constraints += self.m.get_constraints(self.var['X'], self.var['X_robust'], self.var['U'],
                                              self.par['X_last'], self.par['X_robust_last'],
                                              self.par['U_last'])
        constraints +=\
            self.m.add_sub_dynamics_constrains(self.var['X'], self.var['X_sub'], self.var['X_robust'], self.var['U'],
                                               self.par['X_last'], self.par['X_sub_last'], self.par['X_robust_last'],
                                               self.par['U_last'], self.var['nu_sub'], self.var['nu_robust'],
                                               self.par_sub_dynamics_matrix)

        # Dynamics:
        constraints += [
            self.var['X'][:, k + 1] ==
            cvx.reshape(self.par['A_bar'][:, k], (self.m.n_x, self.m.n_x)) @ self.var['X'][:, k]
            + cvx.reshape(self.par['B_bar'][:, k], (self.m.n_x, self.m.n_u)) @ self.var['U'][:, k]
            + cvx.reshape(self.par['C_bar'][:, k], (self.m.n_x, self.m.n_u)) @ self.var['U'][:, k + 1]
            + self.par['z_bar'][:, k]
            + self.var['nu'][:, k]
            for k in range(self.K - 1)
        ]

        # Trust region:
        du = self.var['U'] - self.par['U_last']
        dx = self.var['X'] - self.par['X_last']
        dx_sub = self.var['X_sub'] - self.par['X_sub_last']
        dx_robust = self.var['X_robust'] - self.par['X_robust_last']
        constraints += [cvx.norm(dx, 1) + cvx.norm(dx_sub, 1) + cvx.norm(dx_robust, 1) +\
                        cvx.norm(du, 1) <= self.par['tr_radius']]

        # Objective:
        model_objective = self.m.get_objective(self.var['X'], self.var['X_robust'], self.var['U'],
                                               self.par['X_last'], self.par['U_last'], self.par['X_robust_last'])
        sc_objective = cvx.Minimize(
            self.par['weight_nu'] * cvx.norm(self.var['nu'], 1) +\
            self.par['weight_nu'] * cvx.norm(self.var['nu_sub'], 1) +\
            self.par['weight_nu'] * cvx.norm(self.var['nu_robust'], 1)
        )

        objective = sc_objective if model_objective is None else sc_objective + model_objective
        self.prob = cvx.Problem(objective, constraints)

    def set_parameters(self, **kwargs):
        """
        All parameters have to be filled before calling solve().
        """

        for key in kwargs:
            if key in self.par:
                self.par[key].value = kwargs[key]
            else:
                print(f'Parameter \'{key}\' does not exist.')
        #self.init_problem()
        if self.prob is None:
            self.init_problem()
        else:
            self.m.get_sub_dynamics_constrains(self.par['X_last'], self.par['X_sub_last'], self.par['X_robust_last'],
                                               self.par['U_last'],
                                               self.par_sub_dynamics_matrix)

    def set_problem(self):
        pass


    def print_available_parameters(self):
        print('Parameter names:')
        for key in self.par:
            print(f'\t {key}')
        print('\n')

    def print_available_variables(self):
        print('Variable names:')
        for key in self.var:
            print(f'\t {key}')
        print('\n')

    def get_variable(self, name):
        """
        :param name: Name of the variable.
        :return The value of the variable.
        """

        if name in self.var:
            return self.var[name].value
        else:
            print(f'Variable \'{name}\' does not exist.')
            return None

    def solve(self, **kwargs):
        error = False
        try:
            self.prob.solve(**kwargs)
        except cvx.SolverError:
            error = True

        return error
