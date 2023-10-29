from stlpy.benchmarks.base import BenchmarkScenario
from stlpy.benchmarks.common import inside_circle_formula, make_circle_patch
from stlpy.systems import Unicycle
from mylinear import MyDoubleIntegrator
from mydrake_solver_smooth import MyDrakeSmoothSolver
from myScipyGradientSolver import MyScipyGradientSolver

class NonlinearReachAvoid(BenchmarkScenario):
    r"""
    A 2D mobile robot with unicycle dynamics must
    avoid a circular obstacle (:math:`\mathcal{O}`) before reaching 
    a circular goal (:math:`\mathcal{G}`):

    .. math::

        \varphi = G_{[0,T]} \lnot \mathcal{O} \land F_{[0,T]} \mathcal{G}

    :param goal_center:      a tuple ``(px, py)`` defining the center of the
                             goal region
    :param goal_radius:      a scalar defining the goal radius
    :param obstacle_center:  a tuple ``(px, py)`` defining the center of the
                             obstacle region
    :param obstacle_radius:  a scalar defining the obstacle radius
    :param T:                the time horizon for this scenario.
    """
    def __init__(self, goal_center, goal_radius, obstacle_center, obstacle_radius, T):
        self.goal_center = goal_center
        self.goal_radius = goal_radius

        self.obstacle_center = obstacle_center
        self.obstacle_radius = obstacle_radius

        self.T = T

    def GetSpecification(self):
        # Goal Reaching
        at_goal = inside_circle_formula(self.goal_center, self.goal_radius, 0, 1, 4)

        # Obstacle Avoidance
        at_obstacle_1 = inside_circle_formula(self.obstacle_center[0,:],
                self.obstacle_radius[0], 0, 1, 4)
        not_at_obstacle_1 = at_obstacle_1.negation()

        at_obstacle_2 = inside_circle_formula(self.obstacle_center[1,:],
                self.obstacle_radius[1], 0, 1, 4)
        not_at_obstacle_2 = at_obstacle_2.negation()

        # Put all of the constraints together in one specification
        spec = not_at_obstacle_1.always(0, self.T) & not_at_obstacle_2.always(0, self.T) & at_goal.eventually(0, self.T)

        return spec

    def GetSystem(self):
        sys = MyDoubleIntegrator(2, 0.1)
        return sys

    def add_to_plot(self, ax):
        # Make and add circular patches
        obstacle_1 = make_circle_patch(self.obstacle_center[0,:],
                self.obstacle_radius[0], color='k', alpha=0.5)
        goal_1 = make_circle_patch(self.goal_center, self.goal_radius, 
                color='green', alpha=0.5)

        obstacle_2 = make_circle_patch(self.obstacle_center[1, :],
                self.obstacle_radius[1], color='k', alpha=0.5)

        ax.add_patch(obstacle_1)
        ax.add_patch(obstacle_2)
        ax.add_patch(goal_1)

        # set the field of view
        ax.set_xlim((0,20))
        ax.set_ylim((-5,5))
        ax.set_aspect('equal')


import numpy as np
import matplotlib.pyplot as plt

from stlpy.solvers import *

# Specification Parameters
goal = (10, 0)  # goal center and radius
goal_rad = 0.5
obs = np.array([[3.5, 0.5], [7, -0.5]])     # obstacle center and radius
obs_rad = np.array([1.0, 1.0])
T = 40

# Define the system and specification
scenario = NonlinearReachAvoid(goal, goal_rad, obs, obs_rad, T)
spec = scenario.GetSpecification()
sys = scenario.GetSystem()

# Specify any additional running cost (this helps the numerics in
# a gradient-based method)
Q = np.diag([0,0,0,0])
R = 1e-3*np.eye(2)

# Initial state
x0 = np.array([0.0,0.0,0.0,0.0])

# Choose a solver

solver = MyDrakeSmoothSolver(spec, sys, x0, T, k=2.0)
u_min = np.array([-10,-10])
u_max = np.array([10, 10])
x_min = np.array([-15.0, -1.8, -10,-10])
x_max = np.array([15.0, 1.8, 10, 10])
solver.AddControlBounds(u_min, u_max)
solver.AddStateBounds(x_min, x_max)

#solver = MyScipyGradientSolver(spec, sys, x0, T)

# Set bounds on state and control variables


# Add quadratic running cost (optional)
solver.AddQuadraticCost(Q,R)
#solver.AddGoalConstraints(goal=np.array([10, 0, 0, 0]))

# Solve the optimization problem
x, u, _, _ = solver.Solve()

if x is not None:
    # Plot the solution
    ax = plt.gca()
    scenario.add_to_plot(ax)
    #plt.scatter(*x[:2,:])
    plt.show()

# np.save("x.npy",x[:2,:])
# np.save("u.npy",u)