from stlpy.benchmarks.base import BenchmarkScenario
from stlpy.benchmarks.common import inside_circle_formula, make_circle_patch
from stlpy.systems import Unicycle
from mylinear import MyDoubleIntegrator
from mydrake_solver_smooth import MyDrakeSmoothSolver

class NonlinearMultiTask(BenchmarkScenario):
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
            self.obstacles.append((x,y))

        # Create the (randomly generated) set of targets
        self.targets = []
        for i in range(num_groups):
            target_group = []
            for j in range(targets_per_group):
                x = np.random.uniform(0,9)
                y = np.random.uniform(0,9)
                target_group.append((x,y))
            self.targets.append(target_group)
        
        self.goal = (8.0, 8.0)

        print(self.obstacles)
        print(self.targets)
        self.T = T

    def GetSpecification(self):
        # Specify that we must avoid all obstacles
        obstacle_formulas = []
        for obs in self.obstacles:
            tmp_inside = inside_circle_formula(obs, 1.0, 0, 1, 4)
            obstacle_formulas.append(tmp_inside.negation())
        obstacle_avoidance = obstacle_formulas[0]
        for i in range(1, len(obstacle_formulas)):
            obstacle_avoidance = obstacle_avoidance & obstacle_formulas[i]

        # Specify that for each target group, we need to visit at least one
        # of the targets in that group
        target_group_formulas = []
        for target_group in self.targets:
            group_formulas = []
            for target in target_group:
                group_formulas.append(inside_circle_formula(target, 1.0, 0, 1, 4))
            reach_target_group = group_formulas[0]
            for i in range(1, self.targets_per_group):
                reach_target_group = reach_target_group | group_formulas[i]
            target_group_formulas.append(reach_target_group)

        # Put all of the constraints together in one specification
        specification = obstacle_avoidance.always(0, self.T)
        for reach_target_group in target_group_formulas:
            specification = specification & reach_target_group.eventually(0, self.T)
        goal = inside_circle_formula(self.goal, 1.0, 0, 1, 4)
        specification = specification & goal.eventually(0, self.T)
        return specification

    def GetSystem(self):
        return MyDoubleIntegrator(2, dt=0.1)

    def add_to_plot(self, ax):
        # Add red rectangles for the obstacles
        for obstacle in self.obstacles:
            ax.add_patch(make_circle_patch(obstacle, 1.0, color='k', alpha=0.5, zorder=-1))

        # Use the color cycle to choose the colors of each target group
        # (note that this won't work for more than 10 target groups)
        colors = plt.cm.tab10.colors
        for i, target_group in enumerate(self.targets):
            color = colors[i]
            for target in target_group:
                ax.add_patch(make_circle_patch(target, 1.0, color=color, alpha=0.7, zorder=-1))
        color = colors[i+1]
        ax.add_patch(make_circle_patch(self.goal, 1.0, color=color, alpha=0.7, zorder=-1))
        # set the field of view
        ax.set_xlim((0,10))
        ax.set_ylim((0,10))
        ax.set_aspect('equal')


import numpy as np
import matplotlib.pyplot as plt

from stlpy.solvers import *

# Specification Parameters
num_obstacles = 2
num_groups = 2
targets_per_group = 2
T = 25


seed = 28
count_success = 0
time_list = []
total = 50
while seed < total:
    if seed == 28:
        seed += 1
        continue
    # Define the system and specification
    scenario = NonlinearMultiTask(num_obstacles, num_groups, targets_per_group, T, seed=seed)
    print("seed= ", seed)
    seed += 1
    spec = scenario.GetSpecification()
    sys = scenario.GetSystem()

    # Specify any additional running cost (this helps the numerics in
    # a gradient-based method)
    Q = np.diag([0,0,0,0])
    R = 1e-3*np.eye(2)

    # Initial state
    x0 = np.array([0.0,0.0,0.0, 0.0])

    # Choose a solver

    solver = MyDrakeSmoothSolver(spec, sys, x0, T, k=10.0)
    #solver = ScipyGradientSolver(spec, sys, x0, T)
    # Set bounds on state and control variables
    # u_min = np.array([-10,-10])
    # u_max = np.array([10, 10])
    # x_min = np.array([-5.0, -5.0, -5.0, -5.0])
    # x_max = np.array([10.0, 10.0, 10.0, 10.0])
    # solver.AddControlBounds(u_min, u_max)
    # solver.AddStateBounds(x_min, x_max)

    # Add quadratic running cost (optional)
    solver.AddQuadraticCost(Q,R)
    # solver.AddGoalConstraints(goal=np.array([7.5, 8.5, 0, 0]))

    # Solve the optimization problem
    x,u,rho,solve_time = solver.Solve()
    time_list.append(solve_time)
    if x is not None and rho > 0:
        count_success += 1

print("success rate:", count_success/total)
print("mean time:", sum(time_list)/len(time_list))
