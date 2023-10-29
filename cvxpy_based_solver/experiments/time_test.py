from time import time
import numpy as np

from Discretization import FirstOrderHoldFixTime
from SCproblem import FixTimeSubDynamicSCProblem
from utils import format_line, save_arrays


class SCvxSolverFixTime:
    def __init__(self, m, K, iterations, sigma, tr_radius):
        self.m = m
        self.K = K
        self.iterations = iterations

        # Weight constants
        self.w_nu = 1e5  # virtual control weight
        # initial trust region radius
        self.tr_radius = tr_radius
        # trust region variables
        self.rho_0 = 0.00
        self.rho_1 = 0.1
        self.rho_2 = 0.8
        self.alpha = 1.5
        self.beta = 2.5
        ### motion time
        self.sigma = sigma


    def solve(self):
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
        converged = False
        solve_time_list = []
        compile_time_list = []
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

                start = time()
                # get solution
                new_X = problem.get_variable('X')
                new_U = problem.get_variable('U')
                new_X_sub = problem.get_variable('X_sub')
                new_X_robust = problem.get_variable('X_robust')
                if new_X is None:
                    print("fail to solve")
                    print("robustness = ", X_robust[-1, 0])
                    return (X, X_sub, X_robust, U, X_init, all_X, all_X_sub, solve_time_list, compile_time_list)
                solve_time = problem.prob.solution.attr['solve_time'] + problem.prob.solution.attr['setup_time']
                compile_time = t1_tm - t1_it - solve_time
                solve_time_list.append(solve_time)
                compile_time_list.append(compile_time)
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

        return (X, X_sub, X_robust, U, X_init, all_X, all_X_sub, solve_time_list, compile_time_list)


from time import time

from Discretization import FirstOrderHoldFixTime
from SCproblem import FixTimeSubDynamicSCProblem
from utils import format_line, save_arrays

from Models.double_integral_with_sub_state import DoubleIntegral

import matplotlib.pyplot as plt
from stlpy.benchmarks.common import (inside_rectangle_formula,
                                     outside_rectangle_formula,
                                     make_rectangle_patch)

class Multitask:
    def __init__(self, K, num_obstacles, num_groups, targets_per_group, seed=0):
        self.K = K
        np.random.seed(seed=seed)
        self.targets_per_group = targets_per_group
        # Create the (randomly generated) set of obstacles
        self.obstacles = []
        for i in range(num_obstacles):
            x = np.random.uniform(0, 9)  # keep within workspace
            y = np.random.uniform(0, 9)
            self.obstacles.append((x, x + 2, y, y + 2))

        # Create the (randomly generated) set of targets
        self.targets = []
        for i in range(num_groups):
            target_group = []
            for j in range(targets_per_group):
                x = np.random.uniform(0, 9)
                y = np.random.uniform(0, 9)
                target_group.append((x, x + 1, y, y + 1))
            self.targets.append(target_group)
        # print(self.obstacles)
        # print(self.targets)
        obstacle_formulas = []
        for obs in self.obstacles:
            obstacle_formulas.append(outside_rectangle_formula(obs, 0, 1, 4))
        obstacle_avoidance = obstacle_formulas[0]
        for i in range(1, len(obstacle_formulas)):
            obstacle_avoidance = obstacle_avoidance & obstacle_formulas[i]
        obstacle_avoidance.simplify()
        # Specify that for each target group, we need to visit at least one
        # of the targets in that group
        target_group_formulas = []
        for target_group in self.targets:
            group_formulas = []
            for target in target_group:
                group_formulas.append(inside_rectangle_formula(target, 0, 1, 4))
            reach_target_group = group_formulas[0]
            for i in range(1, self.targets_per_group):
                reach_target_group = reach_target_group | group_formulas[i]
            reach_target_group.simplify()
            target_group_formulas.append(reach_target_group)
        # Put all of the constraints together in one specification
        specification = obstacle_avoidance.always(0, self.K)
        for reach_target_group in target_group_formulas:
            specification = specification & reach_target_group.eventually(0, self.K)
        self.spec = specification
        self.flatten_before_sub(self.spec)

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
    def add_to_plot(self, ax):
        # Add red rectangles for the obstacles
        for obstacle in self.obstacles:
            ax.add_patch(make_rectangle_patch(*obstacle, color='k', alpha=0.5, zorder=-1))

        # Use the color cycle to choose the colors of each target group
        # (note that this won't work for more than 10 target groups)
        colors = plt.cm.tab10.colors
        for i, target_group in enumerate(self.targets):
            color = colors[i]
            for target in target_group:
                ax.add_patch(make_rectangle_patch(*target, color=color, alpha=0.7, zorder=-1))
        color = colors[i+1]
        #ax.add_patch(make_rectangle_patch(*self.goal, color=color, alpha=0.7, zorder=-1))
        # set the field of view
        ax.set_xlim((0, 10))
        ax.set_ylim((0, 10))
        ax.set_aspect('equal')

total_solve_time_list = []
total_compile_time_list = []
step_solve_time_list = []
step_compile_time_list = []
time_step = []
number_of_variable = []

for i in range(15, 151):
    K = i
    iterations = 20
    tr_radius = 30.0
    sigma = K - 1

    either_or = Multitask(K, 1, 5, 2)

    m = DoubleIntegral(K, sigma, either_or.spec, max_k=5, smin_C=0.1,
                       x_init=np.array([2.0, 2.0, 0, 0]), x_final=np.array([7.5, 8.5, 0, 0]))

    m.settingStateBoundary(x_min=np.array([0.0, 0.0, -1.0, -1.0]), x_max=np.array([10.0, 10.0, 1.0, 1.0]))
    m.settingControlBoundary(u_min=np.array([-1, -1]), u_max=np.array([1, 1]))
    m.settingWeights(u_weight=1.0, velocity_weight=0.1)
    solver = SCvxSolverFixTime(m, K, iterations, sigma, tr_radius)

    X, X_sub, X_robust, U, X_init, all_X, all_X_sub, solve_time_list, compile_time_list = solver.solve()
    time_step.append(K - 1)
    number_of_variable.append(m.n_x*K+m.n_x_sub*K+m.n_x_robust)
    total_solve_time_list.append(np.sum(solve_time_list))
    total_compile_time_list.append(np.sum(compile_time_list))
    step_solve_time_list.append(np.mean(solve_time_list))
    step_compile_time_list.append(np.mean(compile_time_list))

    if i % 5 == 0:
        np.save("time_test_result/time_step.npy", np.array(time_step))
        np.save("time_test_result/number_of_variable.npy", np.array(number_of_variable))
        np.save("time_test_result/total_solve_time_list.npy", np.array(total_solve_time_list))
        np.save("time_test_result/total_compile_time_list.npy", np.array(total_compile_time_list))
        np.save("time_test_result/step_solve_time_list.npy", np.array(step_solve_time_list))
        np.save("time_test_result/step_compile_time_list.npy", np.array(step_compile_time_list))

plt.plot(time_step, total_solve_time_list)
plt.xlabel('time step')
plt.ylabel('average solve time')
plt.show()

plt.plot(time_step, total_compile_time_list)
plt.xlabel('time step')
plt.ylabel('average compile time')
plt.show()

plt.plot(number_of_variable, total_solve_time_list)
plt.xlabel('number of variable')
plt.ylabel('average solve time')
plt.show()

plt.plot(number_of_variable, total_compile_time_list)
plt.xlabel('number of variable')
plt.ylabel('average compile time')
plt.show()

