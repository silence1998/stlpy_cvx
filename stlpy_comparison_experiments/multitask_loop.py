import numpy as np
import matplotlib.pyplot as plt

from stlpy.benchmarks.base import BenchmarkScenario
from stlpy.benchmarks.common import (inside_rectangle_formula,
                     outside_rectangle_formula,
                     make_rectangle_patch)
from stlpy.systems import DoubleIntegrator
from stlpy.solvers import *

class RandomMultitarget(BenchmarkScenario):
    r"""
    A 2D mobile robot with double integrator dynamics must 
    navigate through a field of obstacles (grey, :math:`\mathcal{O}_i`)
    and reach at least one target of each color (:math:`\mathcal{T}_i^j`):

    .. math::

        \varphi = 
            \bigwedge_{i=1}^{N_c} \left( \bigvee_{j=1}^{N_t} F_{[0,T]} T_{i}^{j} \right) 
            \land G_{[0,T]} (\bigwedge_{k=1}^{N_o} \lnot O_k),

    :param num_obstacles:       number of obstacles, :math:`N_o`
    :param num_groups:          number of target groups/colors, :math:`N_c`
    :param targets_per_group:   number of targets in each group, :math:`N_t`
    :param T:                   time horizon of the specification
    :param seed:                (optional) seed for random generation of obstacle 
                                and target locations. Default is ``None``.
    """
    def __init__(self, num_obstacles, num_groups, targets_per_group, T, seed=None):
        self.T = T
        self.targets_per_group = targets_per_group

        # Set the seed for the random number generator (for reproducability)
        np.random.seed(seed)

        # Create the (randomly generated) set of obstacles
        self.obstacles = []
        for i in range(num_obstacles):
            x = np.random.uniform(0,9)  # keep within workspace
            y = np.random.uniform(0,9)
            self.obstacles.append((x,x+2,y,y+2))

        # Create the (randomly generated) set of targets
        self.targets = []
        for i in range(num_groups):
            target_group = []
            for j in range(targets_per_group):
                x = np.random.uniform(0,9)
                y = np.random.uniform(0,9)
                target_group.append((x,x+1,y,y+1))
            self.targets.append(target_group)
        
        self.goal = (7.5, 8.5, 7.5, 8.5)
        
        
        print(self.obstacles)
        print(self.targets)
        self.T = T

    def GetSpecification(self):
        # Specify that we must avoid all obstacles
        obstacle_formulas = []
        for obs in self.obstacles:
            obstacle_formulas.append(outside_rectangle_formula(obs, 0, 1, 6))
        obstacle_avoidance = obstacle_formulas[0]
        for i in range(1, len(obstacle_formulas)):
            obstacle_avoidance = obstacle_avoidance & obstacle_formulas[i]

        # Specify that for each target group, we need to visit at least one
        # of the targets in that group
        target_group_formulas = []
        for target_group in self.targets:
            group_formulas = []
            for target in target_group:
                group_formulas.append(inside_rectangle_formula(target, 0, 1, 6))
            reach_target_group = group_formulas[0]
            for i in range(1, self.targets_per_group):
                reach_target_group = reach_target_group | group_formulas[i]
            target_group_formulas.append(reach_target_group)
        goal_area = inside_rectangle_formula(self.goal, 0, 1, 6)
        # Put all of the constraints together in one specification
        specification = obstacle_avoidance.always(0, self.T)
        for reach_target_group in target_group_formulas:
            specification = specification & reach_target_group.eventually(0, self.T)
        specification = specification & goal_area.eventually(0, self.T)
        return specification

    def GetSystem(self):
        return DoubleIntegrator(2)

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
        ax.set_xlim((0,10))
        ax.set_ylim((0,10))
        ax.set_aspect('equal')

# Specification Parameters
num_obstacles = 2
num_groups = 2
targets_per_group = 2
T = 25

seed = 0
count_success = 0
time_list = []
total = 10
while seed < total:
    # Define the specification and system dynamics
    scenario = RandomMultitarget(
            num_obstacles, num_groups, targets_per_group, T, seed=seed)
    seed += 1
    spec = scenario.GetSpecification()
    sys = scenario.GetSystem()

    # Specify any additional running cost (this helps the numerics in
    # a gradient-based method)
    Q = 1e-1*np.diag([0,0,1,1])   # just penalize high velocities
    R = 1e-1*np.eye(2)

    # Initial state
    x0 = np.array([0.0,0.0,0,0])

    # Define the solver
    #solver = GurobiMICPSolver(spec, sys, x0, T, robustness_cost=True)
    solver = DrakeMICPSolver(spec, sys, x0, T, robustness_cost=True)
    #solver = DrakeSos1Solver(spec, sys, x0, T, robustness_cost=True)

    # Set bounds on state and control variables
    u_min = np.array([-0.5,-0.5])
    u_max = np.array([0.5, 0.5])
    x_min = np.array([0.0, 0.0, -1.0, -1.0])
    x_max = np.array([10.0, 10.0, 1.0, 1.0])
    solver.AddControlBounds(u_min, u_max)
    solver.AddStateBounds(x_min, x_max)

    # Add quadratic running cost (optional)
    solver.AddQuadraticCost(0.01*Q,0.01*R)

    # Solve the optimization problem
    x, u, rho, solve_time = solver.Solve()
    time_list.append(solve_time)
    if x is not None and rho > 0:
        count_success += 1

print("success rate:", count_success/total)
print("mean time:", sum(time_list)/len(time_list))

