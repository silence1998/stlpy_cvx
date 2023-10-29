from time import time
import numpy as np

from Discretization import FirstOrderHoldFixTime
from SCproblem import FixTimeSubDynamicSCProblem
from utils import format_line, save_arrays


class SCvxSolverFixTime:
    def __init__(self, m, K, iterations, sigma, tr_radius, w_nu=1e5,
                 rho_1=0.1, rho_2=0.8, alpha=1.5, beta=2.5):
        self.m = m
        self.K = K
        self.iterations = iterations

        # Weight constants
        self.w_nu = w_nu  # virtual control weight
        # initial trust region radius
        self.tr_radius = tr_radius
        # trust region variables
        self.rho_0 = 0.00
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.alpha = alpha
        self.beta = beta
        ### motion time
        self.sigma = sigma
        self.optimal_cost = None
        self.solving_time = None
        self.compile_time = None


    def solve(self):
        self.solving_time = 0.0
        self.compile_time = 0.0
        t_total_1 = time()
        # state and input
        X = np.empty(shape=[self.m.n_x, self.K])
        X_sub = np.empty(shape=[self.m.n_x_sub, self.K])
        X_robust = np.empty(shape=[self.m.n_x_robust, 1])
        U = np.empty(shape=[self.m.n_u, self.K])

        # INITIALIZATION--------------------------------------------------------------------------------------------------------
        X, X_sub, X_robust, U = self.m.initialize_trajectory(X, X_sub, X_robust, U)
        X_init = X
        # START SUCCESSIVE CONVEXIFICATION--------------------------------------------------------------------------------------
        all_X = [X]
        all_X_sub = [X_sub]
        all_U = [U]

        integrator = FirstOrderHoldFixTime(self.m, self.K, self.sigma)
        problem = FixTimeSubDynamicSCProblem(self.m, self.K)

        last_nonlinear_cost = None
        object_value = None
        converged = False
        for it in range(self.iterations):
            t0_it = time()
            print('-' * 50)
            print('-' * 18 + f' Iteration {str(it + 1).zfill(2)} ' + '-' * 18)
            print('-' * 50)

            t0_tm = time()
            A_bar, B_bar, C_bar, z_bar = integrator.calculate_discretization(X, U)
            print(format_line('Time for transition matrices', time() - t0_tm, 's'))

            problem.set_parameters(A_bar=A_bar, B_bar=B_bar, C_bar=C_bar, z_bar=z_bar,
                                   X_last=X, X_sub_last=X_sub, X_robust_last=X_robust, U_last=U,
                                   weight_nu=self.w_nu, tr_radius=self.tr_radius)

            while True:
                t1_it = time()
                error = problem.solve(verbose=False, solver='ECOS', max_iters=200)
                t1_tm = time()
                print(format_line('Solver Error', error))

                # get solution
                new_X = problem.get_variable('X')
                new_U = problem.get_variable('U')
                new_X_sub = problem.get_variable('X_sub')
                new_X_robust = problem.get_variable('X_robust')
                new_object_value = problem.prob.solution.opt_val

                if new_X is None:
                    print("fail to solve")
                    print("robustness = ", X_robust[-1, 0])
                    return (X, X_sub, X_robust, U, X_init, all_X, all_X_sub)
                solve_time = problem.prob.solution.attr['solve_time'] + problem.prob.solution.attr['setup_time']
                compile_time = t1_tm - t1_it - solve_time
                self.solving_time += solve_time
                self.compile_time += compile_time
                # print(format_line('One step Solve time: ', solve_time, 's'))
                # print(format_line('One step Compile time: ', t1_tm - t1_it - solve_time, 's'))

                X_nl = integrator.integrate_nonlinear_piecewise(new_X, new_U)
                X_sub_nl, X_robust_nl = self.m.calculate_subdynamics(X_nl)

                linear_cost_dynamics = np.linalg.norm(problem.get_variable('nu'), 1) + np.linalg.norm(
                    problem.get_variable('nu_sub'), 1) + np.linalg.norm(
                    problem.get_variable('nu_robust'), 1)
                nonlinear_cost_dynamics = np.linalg.norm(new_X - X_nl, 1) + \
                                          np.linalg.norm(new_X_sub - X_sub_nl, 1) + \
                                          np.linalg.norm(new_X_robust - X_robust_nl, 1)
                linear_cost_constraints = self.m.get_linear_cost()
                nonlinear_cost_constraints = self.m.get_nonlinear_cost(X=new_X, U=new_U)

                linear_cost = linear_cost_dynamics + linear_cost_constraints  # J
                nonlinear_cost = nonlinear_cost_dynamics + nonlinear_cost_constraints  # L

                if last_nonlinear_cost is None:
                    last_nonlinear_cost = nonlinear_cost
                    X = new_X
                    U = new_U
                    X_sub = X_sub_nl
                    X_robust = X_robust_nl
                    # X_sub = new_X_sub
                    # X_robust = new_X_robust
                    object_value = new_object_value
                    break

                actual_change = last_nonlinear_cost - nonlinear_cost  # delta_J
                predicted_change = last_nonlinear_cost - linear_cost  # delta_L

                print('')
                print(format_line('Virtual Control Cost', linear_cost_dynamics))
                print(format_line('Constraint Cost', linear_cost_constraints))
                print('')
                print(format_line('Actual change', actual_change))
                print(format_line('Predicted change', predicted_change))
                print('')

                if (abs(predicted_change) < 0.2 and abs(actual_change) < 5e-3) or self.tr_radius < 1e-5:
                    # if abs(predicted_change) < 0.1:
                    converged = True
                    break
                else:
                    rho = actual_change / predicted_change
                    if rho < self.rho_0:
                        # reject solution
                        self.tr_radius /= self.alpha
                        print(f'Trust region too large. Solving again with radius={self.tr_radius}')
                    else:
                        # accept solution
                        X = new_X
                        U = new_U
                        X_sub = X_sub_nl
                        X_robust = X_robust_nl
                        # X_sub = new_X_sub
                        # X_robust = new_X_robust
                        object_value = new_object_value

                        print('Solution accepted.')

                        if rho < self.rho_1:
                            print('Decreasing radius.')
                            self.tr_radius /= self.alpha
                        elif rho >= self.rho_2:
                            print('Increasing radius.')
                            self.tr_radius *= self.beta

                        last_nonlinear_cost = nonlinear_cost
                        break

                problem.set_parameters(tr_radius=self.tr_radius)

                print('-' * 50)

            print('')
            print(format_line('Time for iteration', time() - t0_it, 's'))
            print('')

            all_X.append(X)
            all_X_sub.append(X_sub)
            all_U.append(U)

            if converged:
                print(f'Converged after {it + 1} iterations.')
                break

        if not converged:
            print('Maximum number of iterations reached without convergence.')
        print(format_line('Time for transition matrices', time() - t_total_1, 's'))
        print("robustness = ", X_robust[-1, 0])
        # print("object value = ", object_value)
        self.optimal_cost = object_value
        return (X, X_sub, X_robust, U, X_init, all_X, all_X_sub)


