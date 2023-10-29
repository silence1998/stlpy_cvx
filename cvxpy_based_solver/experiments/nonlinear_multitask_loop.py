from time import time
import numpy as np

from base_task import BaseTask

from Models.double_integral_with_sub_state import DoubleIntegral

import matplotlib.pyplot as plt

from stlpy.benchmarks.common import (inside_circle_formula,
                                     outside_rectangle_formula,
                                     make_circle_patch)
from stlpy.systems.linear import LinearSystem
from stlpy.benchmarks.base import BenchmarkScenario

from SCvx_solver import SCvxSolverFixTime

class Multitask(BaseTask):
    def __init__(self, K, num_obstacles, num_groups, targets_per_group, seed=0):
        BaseTask.__init__(self, K)
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
        # obstacle_avoidance.simplify()
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

    def add_to_plot(self, ax):
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

class MyDoubleIntegrator(LinearSystem):
    def __init__(self, d, dt):
        I = np.eye(d)
        z = np.zeros((d,d))

        A = np.block([[I,I * dt],
                      [z,I]])
        B = np.block([[z],
                      [I * dt]])
        C = np.block([[I,z],
                      [z,I]])
        D = np.block([[z],
                      [z]])

        LinearSystem.__init__(self, A, B, C, D)

class NonlinearMultiTask(BenchmarkScenario): #### benchmark from stlpy used for detection whether the solution trajectory possible
    def __init__(self, num_obstacles, num_groups, targets_per_group, T, seed=None):
        self.T = T
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
        self.T = T

    def GetSpecification(self):
        # Specify that we must avoid all obstacles
        obstacle_formulas = []
        for obs in self.obstacles:
            tmp_inside = inside_circle_formula(obs, 1.0, 0, 1, 4)
            obstacle_formulas.append(tmp_inside.negation())
        obstacle_avoidance = obstacle_formulas[0]
        for i in range(1, len(obstacle_formulas)):
            obstacle_avoidance = obstacle_avoidance & obstacle_formulas[i]

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
        specification = obstacle_avoidance.always(0, self.T)
        for reach_target_group in target_group_formulas:
            specification = specification & reach_target_group.eventually(0, self.T)
        goal = inside_circle_formula(self.goal, 1.0, 0, 1, 4)
        specification = specification & goal.eventually(0, self.T)
        return specification

    def GetSystem(self):
        return MyDoubleIntegrator(2, dt=0.1)

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


K = 26
iterations = 20
sigma = K - 1

num_obstacles = 2
num_groups = 2
targets_per_group = 2

seed = 0
count_success = 0
robustness_list = []
solve_time_list = []
compile_time_list = []
total = 50
while seed < total:
    tr_radius = 80
    Multitask_ = Multitask(K, num_obstacles, num_groups, targets_per_group, seed=seed)
    stlpyMultitask_ = NonlinearMultiTask(num_obstacles, num_groups, targets_per_group, K-1, seed=seed)
    seed += 1
    m = DoubleIntegral(K, sigma, Multitask_.spec, max_k=2, smin_C=0.1,
                       x_init=np.array([0.0, 0.0, 0, 0]), x_final=np.array([8.0, 8.0, 0, 0]))

    m.settingStateBoundary(x_min=np.array([-5.0, -5.0, -5.0, -5.0]), x_max=np.array([10.0, 10.0, 5.0, 5.0]))
    m.settingControlBoundary(u_min=np.array([-5, -5]), u_max=np.array([5, 5]))
    m.settingWeights(u_weight=0.1, velocity_weight=0.0)
    solver = SCvxSolverFixTime(m, K, iterations, sigma, tr_radius)
    t_total_1 = time()
    X, X_sub, X_robust, U, X_init, all_X, all_X_sub = solver.solve()
    X_converge = X
    X_sub_converge = X_sub
    U_converge = U
    robustness_list.append(X_robust[-1, 0])
    time_final = time() - t_total_1
    solve_time_list.append(solver.solving_time)
    compile_time_list.append(solver.compile_time)
    stl_spec = stlpyMultitask_.GetSpecification()
    true_robust = stl_spec.robustness(X, 0)[0]
    print("true_robust: ", true_robust)
    if true_robust > 0:
        count_success += 1
    X_converge = X
    X_sub_converge = X_sub
    U_converge = U
    f = plt.gca()
    f.set_aspect('equal')
    Multitask_.add_to_plot(f)
    # f.scatter(X_converge[0, :], X_converge[1, :])
    # f.scatter(X_init[0, :], X_init[1, :], color='green')
    plt.show()
    # np.save("x2.npy", X_converge[:, :])
    difference_x = []
    for i in range(len(all_X) - 2):
        difference_x.append(np.linalg.norm(all_X[i] - X_converge, 1) + np.linalg.norm(all_X_sub[i] - X_sub_converge, 1))

    difference_x = np.array(difference_x)
    print(difference_x)
    f, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 2e1)
    ax.plot(range(len(difference_x)), difference_x)
    plt.show()

print("success rate:", count_success/total)
print("mean solve time:", sum(solve_time_list)/len(solve_time_list))
print("mean compile time:", sum(compile_time_list)/len(compile_time_list))