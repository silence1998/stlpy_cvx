from stlpy.benchmarks.base import BenchmarkScenario
from stlpy.benchmarks.common import (inside_rectangle_formula,
                                     outside_rectangle_formula,
                                     make_rectangle_patch)
from stlpy.systems import DoubleIntegrator

import numpy as np

from linear_predicate_with_velocity import outside_rectangle_formula_with_velocity

from drake_solver_with_velocity import DrakeMICPSolverWithVelocity


class MovingObstacles(BenchmarkScenario):
    r"""
    A 2D mobile robot with double integrator dynamics must navigate around
    several obstacles (:math:`\mathcal{O}_i`) before reaching one of two
    goals (:math:`\mathcal{G}_i`).

    .. math::

        \varphi = F_{[0,T]}(\mathcal{G}_1 \lor \mathcal{G}_2) \land
            G_{[0,T]} \left( \bigwedge_{i=1}^4 \lnot \mathcal{O}_i \right)

    :param T:   The time horizon of the specification.
    """

    def __init__(self, T, v):
        # Define obstacle and goal regions by their bounds,
        # (xmin, xmax, ymin, ymax)
        self.obstacles = [(2, 5, 4, 6), (5.5, 9, 3.8, 5.7), (4.6, 8, 0.5, 3.5),
                          (2.2, 4.4, 6.4, 11)]
        self.goals = [(7, 8, 8, 9), (9.5, 10.5, 1.5, 2.5)]
        self.T = T
        self.v = v

    def GetSpecification(self):
        # Goal Reaching
        goal_formulas = []
        for goal in self.goals:
            goal_formulas.append(inside_rectangle_formula(goal, 0, 1, 6))

        at_any_goal = goal_formulas[0]
        for i in range(1, len(goal_formulas)):
            at_any_goal = at_any_goal | goal_formulas[i]

        # Obstacle Avoidance
        obstacle_formulas = []
        for obs in self.obstacles:
            obstacle_formulas.append(
                outside_rectangle_formula_with_velocity(obs, 0, 1, self.v, 6))
        ##try : set velocity to [1,0]
        obstacle_avoidance = obstacle_formulas[0]
        for i in range(1, len(obstacle_formulas)):
            obstacle_avoidance = obstacle_avoidance & obstacle_formulas[i]

        # Put all of the constraints together in one specification
        specification = at_any_goal.eventually(0, self.T) & \
                        obstacle_avoidance.always(0, self.T)

        return specification

    def GetSystem(self):
        return DoubleIntegrator(2)

    def add_to_plot(self, ax, t):
        # Make and add rectangular patches
        for obstacle in self.obstacles:
            x1, x2, y1, y2 = obstacle
            x1 += self.v[0] * t
            x2 += self.v[0] * t
            y1 += self.v[1] * t
            y2 += self.v[1] * t
            obstacle = (x1, x2, y1, y2)
            ax.add_patch(make_rectangle_patch(*obstacle, color='k', alpha=0.5))
        for goal in self.goals:
            ax.add_patch(make_rectangle_patch(*goal, color='green', alpha=0.5))

        # set the field of view
        ax.set_xlim((0, 12))
        ax.set_ylim((0, 12))
        ax.set_aspect('equal')


import matplotlib.pyplot as plt

# Specification Parameters
T = 25

# Create the specification and define the dynamics
scenario = MovingObstacles(T, [0, -0.2])
spec = scenario.GetSpecification()
spec.simplify()
sys = scenario.GetSystem()

# Specify any additional running cost (this helps the numerics in
# a gradient-based method)
Q = 1e-1 * np.diag([0, 0, 1, 1])  # just penalize high velocities
R = 1e-1 * np.eye(2)

# Initial state
x0 = np.array([3.0, 3.6, 0, 0])

# Specify a solution method
#solver = GurobiMICPSolver(spec, sys, x0, T, robustness_cost=True)
solver = DrakeMICPSolverWithVelocity(spec, sys, x0, T, robustness_cost=True)
#solver = DrakeSos1Solver(spec, sys, x0, T, robustness_cost=True)

# Set bounds on state and control variables
u_min = np.array([-0.5, -0.5])
u_max = np.array([0.5, 0.5])
x_min = np.array([0.0, 0.0, -1.0, -1.0])
x_max = np.array([10.0, 10.0, 1.0, 1.0])
solver.AddControlBounds(u_min, u_max)
solver.AddStateBounds(x_min, x_max)

# Add quadratic running cost (optional)
solver.AddQuadraticCost(Q, R)

# Solve the optimization problem
x, u, _, _ = solver.Solve()

for t in range(T):
    plt.figure(1)
    plt.clf()
    ax = plt.gca()
    scenario.add_to_plot(ax, t)
    plt.scatter(*x[:2, t])
    plt.plot()
    plt.pause(0.1)
