from time import time
import numpy as np

from Models.double_integral_with_sub_state import DoubleIntegral

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

import stlpy.STL
from stlpy.STL import (LinearPredicate, NonlinearPredicate)
from stlpy.benchmarks.common import (inside_rectangle_formula,
                                     outside_rectangle_formula,
                                     make_rectangle_patch)

from SCvx_solver import SCvxSolverFixTime
import seaborn as sns
import pandas as pd

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

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

map_robustness = np.zeros((9, 9))
map_cost = np.zeros((9, 9))
trajectory_map = np.zeros((9, 9, 4, 25))

list_trust_region = []
list_robustness = []
w_nu_list = [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6]
for j in range(0, 9):
    for i in range(0, 9):
        K = 25
        iterations = 20
        tr_radius = (j + 1) * 10
        sigma = K - 1
        list_trust_region.append(tr_radius)

        num_obstacles = 1
        num_groups = 5
        targets_per_group = 2
        Multitask_ = Multitask(K, num_obstacles, num_groups, targets_per_group, seed=0)

        m = DoubleIntegral(K, sigma, Multitask_.spec, max_k=10, smin_C=0.1,
                           x_init=np.array([5.0, 2.0, 0, 0]), x_final=np.array([0.8, 8.0, 0, 0]))

        m.settingStateBoundary(x_min=np.array([0, 0, -1, -1]), x_max=np.array([10.0, 10.0, 1, 1]))
        m.settingControlBoundary(u_min=np.array([-0.5, -0.5]), u_max=np.array([0.5, 0.5]))
        m.settingWeights(u_weight=1e-2, velocity_weight=1e-2)
        solver = SCvxSolverFixTime(m, K, iterations, sigma, tr_radius, w_nu=w_nu_list[i])

        X, X_sub, X_robust, U, X_init, all_X, all_X_sub = solver.solve()

        #list_robustness.append(X_robust[-1, 0])
        map_robustness[j, i] = X_robust[-1, 0]
        map_cost[j, i] = solver.optimal_cost
        trajectory_map[j, i, :, :] = X

np.save("map_robustness.npy", map_robustness)
np.save("map_cost.npy", map_cost)
np.save("w_nu_list.npy", np.array(w_nu_list))
np.save("list_trust_region.npy", np.array(list_trust_region))
np.save("trajectory_map.npy", trajectory_map)

trajectory_map_simularity = np.zeros((9, 9))
max_robustness_index = np.unravel_index(np.argmax(map_robustness), map_robustness.shape)
max_robustness_trajectory = trajectory_map[max_robustness_index[0], max_robustness_index[1], :, :]
max_robustness_trajectory = max_robustness_trajectory.reshape(-1)
v1_u = unit_vector(max_robustness_trajectory)
for j in range(0, 9):
    for i in range(0, 9):
        tmp = trajectory_map[j, i, :, :]
        tmp = tmp.reshape(-1)
        v2_u = unit_vector(tmp)
        trajectory_map_simularity[j, i] = np.arccos(np.dot(v1_u, v2_u))

np.save("trajectory_map_simularity.npy", trajectory_map_simularity)

# map_ = np.load("map.npy", allow_pickle=True)
# w_nu_list = np.load("w_nu_list.npy")
# list_trust_region = np.load("list_trust_region.npy")
# map_ = map_[0:8, 0:8]
# print(map_)
# column_names = list_trust_region[0:8]
# w_nu_list_ = ['1e2', '5e2', '1e3', '5e3', '1e4', '5e4', '1e5', '5e5', '1e6']
# row_indices = w_nu_list_[0: 8]
# data_df = pd.DataFrame(map_, index=row_indices, columns=column_names)
# f, ax = plt.subplots()
# #ax.set_yscale("log")
# # ax.imshow(-map_[0:-1, :], cmap='hot', interpolation='nearest')
# # plt.show()
# ax = sns.heatmap(data_df, vmin=0, vmax=0.2)
# plt.xlabel('initial trust region')
# plt.ylabel('penalty coefficient')
# plt.title('robustness')
# plt.show()

