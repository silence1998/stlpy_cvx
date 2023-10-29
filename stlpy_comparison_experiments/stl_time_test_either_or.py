#!/usr/bin/env python

##
#
# Set up, solve, and plot the solution for the "reach-avoid"
# scenario, where the robot must reach one of two targets before
# reaching the goal.
#
##

import numpy as np
import matplotlib.pyplot as plt
from stlpy.solvers import *

from stlpy.benchmarks.base import BenchmarkScenario
from stlpy.benchmarks.common import (inside_rectangle_formula,
                     outside_rectangle_formula,
                     make_rectangle_patch)
from stlpy.systems import DoubleIntegrator

class EitherOr(BenchmarkScenario):
    r"""
    A 2D mobile robot with double integrator dynamics must
    avoid an obstacle (:math:`\mathcal{O}`) before reaching a goal
    (:math:`\mathcal{G}`). Along the way, the robot must reach one
    of two intermediate targets (:math:`\mathcal{T}_i`) and stay
    there for several timesteps:

    .. math::

        \varphi = 
            F_{[0,T-\tau]} 
                \left( G_{[0,\tau]} \mathcal{T}_1 \lor G_{[0,\tau]} \mathcal{T}_2 \right)
            \land F_{[0,T]} \mathcal{G} 
            \land G_{[0,T]} \lnot \mathcal{O}

    :param goal:        Tuple containing bounds of the rectangular goal region
    :param target_one:  Tuple containing bounds of the rectangular first target
    :param target_two:  Tuple containing bounds of the rectangular second target
    :param obstacle:    Tuple containing bounds of the rectangular obstacle
    :param T:           Total number of time steps
    :param T_dwell:     Dwell time :math:`\tau` (integer number of timesteps)
    """
    def __init__(self, goal, target_one, target_two, obstacle, T, T_dwell):
        self.goal = goal
        self.target_one = target_one
        self.target_two = target_two
        self.obstacle = obstacle
        self.T = T
        self.T_dwell = T_dwell

    def GetSpecification(self):
        # Goal Reaching
        at_goal = inside_rectangle_formula(self.goal, 0, 1, 6)

        # Target reaching
        at_target_one = inside_rectangle_formula(self.target_one, 0, 1, 6)#.always(0, self.T_dwell)
        at_target_two = inside_rectangle_formula(self.target_two, 0, 1, 6)#.always(0, self.T_dwell)
        at_either_target = at_target_one | at_target_two

        # Obstacle Avoidance
        not_at_obstacle = outside_rectangle_formula(self.obstacle, 0, 1, 6)

        specification = at_either_target.eventually(0, self.T) & \
                        not_at_obstacle.always(0, self.T) & \
                        at_goal.eventually(0, self.T)

        return specification

    def GetSystem(self):
        return DoubleIntegrator(2)

    def add_to_plot(self, ax):
        # Make and add rectangular patches
        ax.add_patch(make_rectangle_patch(*self.obstacle, color='k', alpha=0.5))
        ax.add_patch(make_rectangle_patch(*self.target_one, color='blue', alpha=0.5))
        ax.add_patch(make_rectangle_patch(*self.target_two, color='blue', alpha=0.5))
        ax.add_patch(make_rectangle_patch(*self.goal, color='green', alpha=0.5))

        # set the field of view
        ax.set_xlim((0,10))
        ax.set_ylim((0,10))
        ax.set_aspect('equal')


# Specification Parameters
goal = (7,8,8,9)     # (xmin, xmax, ymin, ymax)
target_one = (1,2,6,7)
target_two = (7,8,4.5,5.5)
obstacle = (3,5,4,6)
time_step = []
number_of_variable = []
total_solve_time_list = []
for i in range(15, 81):
    T = i
    dwell_time = 10

    # Create the specification
    scenario = EitherOr(goal, target_one, target_two, obstacle, T, dwell_time)
    spec = scenario.GetSpecification()
    spec.simplify()
    sys = scenario.GetSystem()

    # Specify any additional running cost
    Q = 1e-1*np.diag([0,0,1,1])   # just penalize high velocities
    R = 1e-0*np.eye(2)

    # Initial state
    x0 = np.array([2.0,2.0,0,0])

    # Specify a solution strategy
    #solver = GurobiMICPSolver(spec, sys, x0, T, robustness_cost=True)
    #solver = DrakeMICPSolver(spec, sys, x0, T, robustness_cost=True)
    solver = DrakeSos1Solver(spec, sys, x0, T, robustness_cost=True)

    # Set bounds on state and control variables
    u_min = np.array([-0.5,-0.5])
    u_max = np.array([0.5, 0.5])
    x_min = np.array([0.0, 0.0, -1.0, -1.0])
    x_max = np.array([10.0, 10.0, 1.0, 1.0])
    solver.AddControlBounds(u_min, u_max)
    solver.AddStateBounds(x_min, x_max)

    x_init=x0
    x_final=np.array([7.5, 8.5, 0, 0])
    # Add quadratic running cost (optional)
    solver.AddQuadraticCost(Q,R)

    # Solve the optimization problem
    x, u, rho, solve_time = solver.Solve()
    tt = solver.mp.num_vars()
    time_step.append(T)
    number_of_variable.append(solver.mp.num_vars())
    total_solve_time_list.append(solve_time)
    if i % 5 == 0:
        np.save("time_test_result/time_step.npy", np.array(time_step))
        np.save("time_test_result/number_of_variable.npy", np.array(number_of_variable))
        np.save("time_test_result/total_solve_time_list.npy", np.array(total_solve_time_list))

np.save("time_test_result/time_step.npy", np.array(time_step))
np.save("time_test_result/number_of_variable.npy", np.array(number_of_variable))
np.save("time_test_result/total_solve_time_list.npy", np.array(total_solve_time_list))
