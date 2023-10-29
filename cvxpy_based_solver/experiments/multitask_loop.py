from time import time
import numpy as np

from base_task import BaseTask

from Models.double_integral_with_sub_state import DoubleIntegral

import matplotlib.pyplot as plt

from stlpy.benchmarks.common import (inside_rectangle_formula,
                                     outside_rectangle_formula,
                                     make_rectangle_patch)
from stlpy.benchmarks.random_multitarget import RandomMultitarget

from SCvx_solver import SCvxSolverFixTime

class Multitask(BaseTask):
    def __init__(self, K, num_obstacles, num_groups, targets_per_group, seed=0):
        BaseTask.__init__(self, K)
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
        self.goal = (7.5, 8.5, 7.5, 8.5)
        goal_area = inside_rectangle_formula((7.5, 8.5, 7.5, 8.5), 0, 1, 4)
        self.spec = specification & goal_area.eventually(0, self.K)
        self.flatten_before_sub(self.spec)

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
        ax.add_patch(make_rectangle_patch(*self.goal, color=color, alpha=0.7, zorder=-1))
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
    stlpyMultitask_ = RandomMultitarget(num_obstacles, num_groups, targets_per_group, K-1, seed=seed)
    seed += 1
    m = DoubleIntegral(K, sigma, Multitask_.spec, max_k=10, smin_C=0.1,
                       x_init=np.array([0.0, 0.0, 0, 0]), x_final=np.array([8.0, 8.0, 0, 0]))

    m.settingStateBoundary(x_min=np.array([0.0, 0.0, -1.0, -1.0]), x_max=np.array([10.0, 10.0, 1.0, 1.0]))
    m.settingControlBoundary(u_min=np.array([-1.0, -1.0]), u_max=np.array([1.0, 1.0]))
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
    tmp_y = np.vstack((X, U))
    true_robust = stl_spec.robustness(tmp_y, 0)[0]
    print("true_robust: ", true_robust)
    if true_robust > 0:
        count_success += 1

print("success rate:", count_success/total)
print("mean solve time:", sum(solve_time_list)/len(solve_time_list))
print("mean compile time:", sum(compile_time_list)/len(compile_time_list))

