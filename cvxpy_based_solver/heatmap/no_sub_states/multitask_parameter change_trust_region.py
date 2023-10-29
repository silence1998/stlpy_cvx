from time import time
import numpy as np

from Discretization import FirstOrderHoldFixTime
from SCproblem import FixTimeSCProblem
from utils import format_line, save_arrays

from Models.double_integral import DoubleIntegral

map_robustness = np.zeros((9, 9))
map_cost = np.zeros((9, 9))

list_trust_region = []
list_robustness = []
w_nu_list = [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6]
for j in range(0, 9):
    for i in range(0, 9):
        K = 21
        iterations = 50

        solver = ['ECOS', 'MOSEK'][0]
        verbose_solver = False

        # Weight constants
        w_nu = w_nu_list[i]  # virtual control
        w_sigma = 10  # flight time
        # initial trust region radius
        tr_radius = j+1
        # trust region variables
        rho_0 = 0.0
        rho_1 = 0.25
        rho_2 = 0.9
        alpha = 2.0
        beta = 3.2

        sigma = 15.0
        m = DoubleIntegral(K, sigma)

        # state and input
        X = np.empty(shape=[m.n_x, K])
        U = np.empty(shape=[m.n_u, K])

        # INITIALIZATION--------------------------------------------------------------------------------------------------------
        X, U = m.initialize_trajectory(X, U)

        # START SUCCESSIVE CONVEXIFICATION--------------------------------------------------------------------------------------
        all_X = X
        all_U = U

        integrator = FirstOrderHoldFixTime(m, K, sigma)
        problem = FixTimeSCProblem(m, K)

        last_nonlinear_cost = None
        converged = False
        object_value = 0
        for it in range(iterations):
            t0_it = time()
            print('-' * 50)
            print('-' * 18 + f' Iteration {str(it + 1).zfill(2)} ' + '-' * 18)
            print('-' * 50)

            t0_tm = time()
            A_bar, B_bar, C_bar, z_bar = integrator.calculate_discretization(X, U)
            print(format_line('Time for transition matrices', time() - t0_tm, 's'))

            problem.set_parameters(A_bar=A_bar, B_bar=B_bar, C_bar=C_bar, z_bar=z_bar,
                                   X_last=X, U_last=U,
                                   weight_nu=w_nu, tr_radius=tr_radius)

            while True:
                error = problem.solve(verbose=verbose_solver, solver=solver, max_iters=200)
                print(format_line('Solver Error', error))

                # get solution
                new_X = problem.get_variable('X')
                new_U = problem.get_variable('U')
                new_object_value = problem.prob.solution.opt_val

                X_nl = integrator.integrate_nonlinear_piecewise(new_X, new_U)

                linear_cost_dynamics = np.linalg.norm(problem.get_variable('nu'), 1)
                nonlinear_cost_dynamics = np.linalg.norm(new_X - X_nl,
                                                         1)  #### why integete to get X_nl, in paper it is later one minus previous one

                linear_cost_constraints = m.get_linear_cost()
                nonlinear_cost_constraints = m.get_nonlinear_cost(X=new_X, U=new_U)

                linear_cost = linear_cost_dynamics + linear_cost_constraints  # J
                nonlinear_cost = nonlinear_cost_dynamics + nonlinear_cost_constraints  # L

                if last_nonlinear_cost is None:
                    last_nonlinear_cost = nonlinear_cost
                    X = new_X
                    U = new_U
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
                print(format_line('Final time', sigma))
                print('')

                if abs(predicted_change) < 1e-4:
                    converged = True
                    break
                else:
                    rho = actual_change / predicted_change
                    if rho < rho_0:
                        # reject solution
                        tr_radius /= alpha
                        print(f'Trust region too large. Solving again with radius={tr_radius}')
                    else:
                        # accept solution
                        X = new_X
                        U = new_U
                        object_value = new_object_value

                        print('Solution accepted.')

                        if rho < rho_1:
                            print('Decreasing radius.')
                            tr_radius /= alpha
                        elif rho >= rho_2:
                            print('Increasing radius.')
                            tr_radius *= beta

                        last_nonlinear_cost = nonlinear_cost
                        break

                problem.set_parameters(tr_radius=tr_radius)

                print('-' * 50)

            print('')
            print(format_line('Time for iteration', time() - t0_it, 's'))
            print('')

            all_X = X
            all_U = U

            if converged:
                print(f'Converged after {it + 1} iterations.')
                break

        if not converged:
            print('Maximum number of iterations reached without convergence.')
        map_cost[j, i] = object_value

np.save("map_cost.npy", map_cost)
np.save("w_nu_list.npy", np.array(w_nu_list))
np.save("list_trust_region.npy", np.array(list_trust_region))

