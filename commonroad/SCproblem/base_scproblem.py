import cvxpy as cvx
from abc import ABC, abstractmethod

class SCProblem(ABC):
    """
    Defines a standard Successive Convexification problem and adds the model specific constraints and objectives.

    :param m: The model object
    :param K: Number of discretization points
    """

    @abstractmethod
    def set_parameters(self, **kwargs):
        """
        All parameters have to be filled before calling solve().
        """

        for key in kwargs:
            if key in self.par:
                self.par[key].value = kwargs[key]
            else:
                print(f'Parameter \'{key}\' does not exist.')
        self.set_problem()

    @abstractmethod
    def set_problem(self):
        """
        no return,
        create cost function and constrains and generate cvxproblem
        """
        pass

    @abstractmethod
    def print_available_parameters(self):
        print('Parameter names:')
        for key in self.par:
            print(f'\t {key}')
        print('\n')

    @abstractmethod
    def print_available_variables(self):
        print('Variable names:')
        for key in self.var:
            print(f'\t {key}')
        print('\n')

    @abstractmethod
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

    @abstractmethod
    def solve(self, **kwargs):
        error = False
        try:
            self.prob.solve(**kwargs)
        except cvx.SolverError:
            error = True

        return error
