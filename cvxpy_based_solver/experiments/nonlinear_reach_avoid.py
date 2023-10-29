from time import time
import numpy as np

from base_task import BaseTask

from Models.double_integral_with_sub_state import DoubleIntegral
from stlpy.benchmarks.common import inside_circle_formula, make_circle_patch

import matplotlib.pyplot as plt

from SCvx_solver import SCvxSolverFixTime

class Avoid(BaseTask):
    def __init__(self, K):
        BaseTask.__init__(self, K)
        self.x_ob = np.array([1.99, 2.0])

        # Obstacle Avoidance
        at_obstacle = inside_circle_formula(self.x_ob, 1.0, 0, 1, 4)
        not_at_obstacle = at_obstacle.negation()

        self.spec = not_at_obstacle.always(0, self.K)
        # self.spec = not_at_obstacle.always(0, self.K) & \
        #              at_goal.eventually(0, self.K)
        # self.spec = at_goal.eventually(0, self.K)
        self.flatten_before_sub(self.spec)

    def add_to_plot(self, ax):
        obstacle_1 = make_circle_patch(self.x_ob,
                          1.0, color='k', alpha=0.5)
        ax.add_patch(obstacle_1)
        # set the field of view
        ax.set_xlim((0, 5))
        ax.set_ylim((0, 5))
        ax.set_aspect('equal')


t_total_1 = time()
K = 26
iterations = 20

# initial trust region radius
tr_radius = 10

sigma = 8.0

avoid = Avoid(K)

m = DoubleIntegral(K, sigma, avoid.spec, max_k=10, smin_C=0.1,
                       x_init=np.array([0.0, 0.0, 0, 0]), x_final=np.array([4.0, 4.0, 0, 0]))
m.settingStateBoundary(x_min=np.array([0.0, 0.0, -1.0, -1.0]), x_max=np.array([10.0, 10.0, 1.0, 1.0]))
m.settingControlBoundary(u_min=np.array([-0.5, -0.5]), u_max=np.array([0.5, 0.5]))
m.settingFixFinalState()
m.settingWeights(u_weight=0.1, velocity_weight=0.0)
solver = SCvxSolverFixTime(m, K, iterations, sigma, tr_radius)
X, X_sub, X_robust, U, X_init, all_X, all_X_sub = solver.solve()
print('Solve time: ', solver.solving_time, 's')
print('Compile time: ', solver.compile_time, 's')
print('final robustness:%f', X_robust[-1, 0])
X_converge = X
X_sub_converge = X_sub
U_converge = U
f = plt.gca()
f.set_aspect('equal')
avoid.add_to_plot(f)
f.scatter(X_converge[0, :], X_converge[1, :], zorder=10)
f.scatter(X_init[0, :], X_init[1, :], color='green')
plt.show()

# difference_x = []
# for i in range(len(all_X) - 2):
#     difference_x.append(np.linalg.norm(all_X[i] - X_converge, 1) + np.linalg.norm(all_X_sub[i] - X_sub_converge, 1))
#
# difference_x = np.array(difference_x)
# print(difference_x)
# f, ax = plt.subplots()
# ax.set_yscale("log")
# ax.set_ylim(1e-4, 2e1)
# ax.plot(range(len(difference_x)), difference_x)
# plt.show()
