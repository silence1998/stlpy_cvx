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
from stlpy.benchmarks import RandomMultitarget


# Specification Parameters

time_step = []
number_of_variable = []
total_solve_time_list = []
for i in range(15, 36):
    T = i

    # Create the specification
    scenario = RandomMultitarget(1, 5, 2, T, seed=0)
    spec = scenario.GetSpecification()
    spec.simplify()
    sys = scenario.GetSystem()

    # Specify any additional running cost
    Q = np.diag([0,0,1,1])   # just penalize high velocities
    R = np.eye(2)

    # Initial state
    x0 = np.array([2.0,2.0,0,0])

    # Specify a solution strategy
    #solver = GurobiMICPSolver(spec, sys, x0, T, robustness_cost=True)
    solver = DrakeMICPSolver(spec, sys, x0, T, robustness_cost=True)
    #solver = DrakeSos1Solver(spec, sys, x0, T, robustness_cost=True)

    # Set bounds on state and control variables
    u_min = np.array([-1.0,-1.0])
    u_max = np.array([1.0, 1.0])
    x_min = np.array([0.0, 0.0, -1, -1])
    x_max = np.array([10.0, 10.0, 1, 1])
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
