import numpy as np

from base_task import BaseTask

from Models.double_integral_with_sub_state import DoubleIntegral

import matplotlib.pyplot as plt

from stlpy.benchmarks.common import (inside_rectangle_formula,
                                     outside_rectangle_formula,
                                     make_rectangle_patch)

from SCvx_solver import SCvxSolverFixTime

class EitherOr(BaseTask):
    def __init__(self, K):
        BaseTask.__init__(self, K)
        goal = (7, 8, 8, 9)  # (xmin, xmax, ymin, ymax)
        target_one = (1, 2, 6, 7)
        target_two = (7, 8, 4.5, 5.5)
        obstacle = (3, 5, 4, 6)
        self.goal = goal
        self.target_one = target_one
        self.target_two = target_two
        self.obstacle = obstacle

        # Goal Reaching
        at_goal = inside_rectangle_formula(self.goal, 0, 1, 4)

        # Target reaching
        at_target_one = inside_rectangle_formula(self.target_one, 0, 1, 4)
        at_target_two = inside_rectangle_formula(self.target_two, 0, 1, 4)
        at_either_target = at_target_one | at_target_two

        # Obstacle Avoidance
        not_at_obstacle = outside_rectangle_formula(self.obstacle, 0, 1, 4)

        at_either_target.simplify()
        not_at_obstacle.simplify()
        at_goal.simplify()

        self.spec = at_either_target.eventually(0, self.K) & \
                    not_at_obstacle.always(0, self.K) & \
                    at_goal.eventually(0, self.K)
        # self.spec = not_at_obstacle.always(0, self.K) & \
        #              at_goal.eventually(0, self.K)
        # self.spec = at_goal.eventually(0, self.K)
        self.flatten_before_sub(self.spec)

    def add_to_plot(self, ax):
        ax.add_patch(make_rectangle_patch(*self.obstacle, color='k', alpha=0.5))
        ax.add_patch(make_rectangle_patch(*self.target_one, color='blue', alpha=0.5))
        ax.add_patch(make_rectangle_patch(*self.target_two, color='blue', alpha=0.5))
        ax.add_patch(make_rectangle_patch(*self.goal, color='green', alpha=0.5))

        # set the field of view
        ax.set_xlim((0, 10))
        ax.set_ylim((0, 10))
        ax.set_aspect('equal')

if __name__ == '__main__':
    K = 26
    iterations = 20
    tr_radius = 30.0  ##### 40 for K=51
    sigma = K - 1

    either_or = EitherOr(K)

    m = DoubleIntegral(K, sigma, either_or.spec, max_k=10, smin_C=0.1,
                       x_init=np.array([2.0, 2.0, 0, 0]), x_final=np.array([7.5, 8.5, 0, 0]))
    m.settingStateBoundary(x_min=np.array([0.0, 0.0, -1.0, -1.0]), x_max=np.array([10.0, 10.0, 1.0, 1.0]))
    m.settingControlBoundary(u_min=np.array([-0.5, -0.5]), u_max=np.array([0.5, 0.5]))
    m.settingWeights(u_weight=1.0, velocity_weight=0.1)
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
    print("robustness: ", X_robust[-1, 0])
    f = plt.gca()
    f.set_aspect('equal')
    either_or.add_to_plot(f)
    f.scatter(X_converge[0, :], X_converge[1, :])
    f.scatter(X_init[0, :], X_init[1, :], color='green')
    plt.show()
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
