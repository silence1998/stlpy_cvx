import numpy as np

from base_task import BaseTask

from Models.double_integral_with_sub_state import DoubleIntegral

import matplotlib.pyplot as plt

from stlpy.benchmarks.common import (inside_rectangle_formula,
                                     outside_rectangle_formula,
                                     make_rectangle_patch)

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
        self.spec = specification
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
        #ax.add_patch(make_rectangle_patch(*self.goal, color=color, alpha=0.7, zorder=-1))
        # set the field of view
        ax.set_xlim((0, 10))
        ax.set_ylim((0, 10))
        ax.set_aspect('equal')

if __name__ == '__main__':
    K = 26
    iterations = 20
    tr_radius = 54 ### 55 for K = 36
    sigma = K - 1


    num_obstacles = 1
    num_groups = 5
    targets_per_group = 2
    Multitask_ = Multitask(K, num_obstacles, num_groups, targets_per_group, seed=0)

    m = DoubleIntegral(K, sigma, Multitask_.spec, max_k=10, smin_C=0.1,
                       x_init=np.array([5.0, 2.0, 0, 0]), x_final=np.array([0.8, 8.0, 0, 0]))

    m.settingStateBoundary(x_min=np.array([0, 0, -1, -1]), x_max=np.array([10.0, 10.0, 1, 1]))
    m.settingControlBoundary(u_min=np.array([-0.5, -0.5]), u_max=np.array([0.5, 0.5]))
    m.settingWeights(u_weight=1e-2, velocity_weight=1e-2)
    solver = SCvxSolverFixTime(m, K, iterations, sigma, tr_radius)

    X, X_sub, X_robust, U, X_init, all_X, all_X_sub = solver.solve()
    print('Solve time: ', solver.solving_time, 's')
    print('Compile time: ', solver.compile_time, 's')
    # np.save("X_multitask.npy", X)
    # np.save("X_sub_multitask.npy", X_sub)
    # np.save("X_robust_multitask.npy", X_robust)
    # np.save("U_multitask.npy", U)
    X_converge = X
    X_sub_converge = X_sub
    U_converge = U
    f = plt.gca()
    f.set_aspect('equal')
    Multitask_.add_to_plot(f)
    f.scatter(X_converge[0, :], X_converge[1, :])
    f.scatter(X_init[0, :], X_init[1, :], color='green')
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
