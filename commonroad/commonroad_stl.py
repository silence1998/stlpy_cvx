import copy
import math
import numpy as np
import sympy as sp
from stlpy.STL import LinearPredicate, NonlinearPredicate
from Models.base_model import MovingNonlinearPredicate

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonroad_dc.feasibility.vehicle_dynamics import KinematicSingleTrackDynamics
from commonroad.scenario.state import KSState, InitialState
from commonroad.scenario.trajectory import Trajectory
from commonroad.common.solution import Solution, PlanningProblemSolution, \
    VehicleModel, VehicleType, CostFunction
from typing import Union, List, Set, Dict, Tuple, overload
from commonroad.scenario.obstacle import StaticObstacle, DynamicObstacle, EnvironmentObstacle, Obstacle, \
    PhantomObstacle
from commonroad_dc.feasibility.solution_checker import valid_solution

############
#self car parameters ###TODO dynamic parameter table
####################
Length = 4.508
Width = 1.610


def circle_formula(center, radius, y1_index, y2_index, y3_index, d, deviation=np.array([0, 0]), name=None):
    def g(y):
        x0 = y[y1_index]
        y0 = y[y2_index]
        angle = y[y3_index]
        if isinstance(y, np.ndarray):
            x = x0+deviation[0]*np.cos(angle)-deviation[1]*np.sin(angle)
            y = y0+deviation[0]*np.sin(angle)+deviation[1]*np.cos(angle)
            return radius - np.sqrt((x-center[0])**2 + (y-center[1])**2+1e-8)
        else:
            x = x0+deviation[0]*sp.cos(angle)-deviation[1]*sp.sin(angle)
            y = y0+deviation[0]*sp.sin(angle)+deviation[1]*sp.cos(angle)
            return radius - sp.sqrt((x-center[0])**2 + (y-center[1])**2+1e-8)
    return NonlinearPredicate(g, d, name=name)

def moving_circle_formula(center_position, d_obs, orientation, radius,
                         y_index_list, center_index_list,
                          d, deviation=np.array([0, 0]), name=None):
    def g(y):
        x0 = y[y_index_list[0]]
        y0 = y[y_index_list[1]]
        angle = y[y_index_list[2]]
        centerx = y[center_index_list[0]]
        centery = y[center_index_list[1]]
        center_angle = y[center_index_list[2]]
        if isinstance(y, np.ndarray):
            x = x0+deviation[0]*np.cos(angle)-deviation[1]*np.sin(angle)
            y = y0+deviation[0]*np.sin(angle)+deviation[1]*np.cos(angle)
            center_x = centerx + d_obs*np.cos(center_angle)
            center_y = centery + d_obs*np.sin(center_angle)
            return radius - np.sqrt((x-center_x)**2 + (y-center_y)**2+1e-8)
        else:
            x = x0+deviation[0]*sp.cos(angle)-deviation[1]*sp.sin(angle)
            y = y0+deviation[0]*sp.sin(angle)+deviation[1]*sp.cos(angle)
            center_x = centerx + d_obs*sp.cos(center_angle)
            center_y = centery + d_obs*sp.sin(center_angle)
            return radius - sp.sqrt((x-center_x)**2 + (y-center_y)**2+1e-8)
    return MovingNonlinearPredicate(g, d, np.vstack((center_position, orientation)), name=name)


def two_rectangle_outside_formula(box, orientation, y1_index, y2_index, y3_index, d, b=0, name=None):
    assert y1_index < d, "index must be less than signal dimension"
    assert y2_index < d, "index must be less than signal dimension"
    assert y3_index < d, "index must be less than signal dimension"

    if orientation < 0:
        return two_rectangle_outside_formula(box, orientation + math.pi,
                                             y1_index, y2_index, y3_index,
                                             d, b, name)
    elif orientation >= math.pi:
        return two_rectangle_outside_formula(box, orientation - math.pi,
                                             y1_index, y2_index, y3_index,
                                             d, b, name)
    elif orientation >= math.pi/2:
        position, length, width = box
        return two_rectangle_outside_formula((position, width, length), orientation - math.pi/2,
                                             y1_index, y2_index, y3_index,
                                             d, b, name)

    position, length, width = box

    ####### divide rectangle into 3 circles
    radius = math.sqrt(length**2/(6**2)+width**2/4)
    d_obs = 2*math.sqrt(radius**2-width**2/4)
    center = []
    center_1 = (position[0], position[1])
    center_2 = (position[0] + d_obs*math.cos(orientation), position[1] + d_obs*math.sin(orientation))
    center_3 = (position[0] - d_obs*math.cos(orientation), position[1] - d_obs*math.sin(orientation))
    center.append(center_1)
    center.append(center_2)
    center.append(center_3)

    radius_car = math.sqrt(Length**2/(6**2)+Width**2/4)
    deviation_car = 2*math.sqrt(radius_car**2-Width**2/4)
    formula_list = []
    for i in range(3):
        for j in [b-deviation_car,b,b+deviation_car]:
            # tmp = circle_formula(center[i], width/2+Width/2,
            #                                    y1_index,y2_index,y3_index,d,[j*deviation_car, 0])
            tmp = circle_formula(center[i], radius+radius_car,
                                               y1_index,y2_index,y3_index,d,[j, 0])
            tmp = tmp.negation()
            formula_list.append(tmp)
    result = formula_list[0]
    for i in range(1, len(formula_list)):
        result = result & formula_list[i]
    result.simplify()
    return result

def two_moving_rectangle_outside_formula(box, orientation, y_index_list, center_index_list, d,
                                         b=0, redundant_radius = 0.0, name=None):
    y1_index, y2_index, y3_index = y_index_list
    
    assert y1_index < d, "index must be less than signal dimension"
    assert y2_index < d, "index must be less than signal dimension"
    assert y3_index < d, "index must be less than signal dimension"

    position, length, width = box

    ####### divide rectangle into 3 circles
    radius = math.sqrt(length**2/(6**2)+width**2/4)
    d_obs = 2*math.sqrt(radius**2-width**2/4)

    radius_car = math.sqrt(Length**2/(6**2)+Width**2/4)
    deviation_car = 2*math.sqrt(radius_car**2-Width**2/4)
    formula_list = []
    for i in [-1,0,1]:
        for j in [b-deviation_car,b,b+deviation_car]:
            tmp = moving_circle_formula(position, i*d_obs, orientation,
                                        radius+radius_car+redundant_radius,
                                        y_index_list, center_index_list, d,[j, 0])
            tmp = tmp.negation()
            formula_list.append(tmp)
    result = formula_list[0]
    for i in range(1, len(formula_list)):
        result = result & formula_list[i]
    result.simplify()
    return result


def line_angle_formula(a, b, y1_index, y2_index, y3_index, d, deviation=np.array([0, 0]), name=None):
    def g(y):
        x0 = y[y1_index]
        y0 = y[y2_index]
        angle = y[y3_index]
        if isinstance(y, np.ndarray):
            x = x0+deviation[0]*np.cos(angle)-deviation[1]*np.sin(angle)
            y = y0+deviation[0]*np.sin(angle)+deviation[1]*np.cos(angle)
            return a[0, 0]*x+a[0, 1]*y-b
        else:
            x = x0+deviation[0]*sp.cos(angle)-deviation[1]*sp.sin(angle)
            y = y0+deviation[0]*sp.sin(angle)+deviation[1]*sp.cos(angle)
            return a[0, 0]*x+a[0, 1]*y-b
    return NonlinearPredicate(g, d, name=name)


def outside_rectangle_deviation(box, orientation, y1_index, y2_index, y3_index, d,
                                     deviation=np.array([0, 0]), name=None): 
    ### a try for 4 vertices outside rectangle
    assert y1_index < d, "index must be less than signal dimension"
    assert y2_index < d, "index must be less than signal dimension"
    assert y3_index < d, "index must be less than signal dimension"
    if orientation < 0:
        return outside_rectangle_deviation(box, orientation+math.pi,
                                           y1_index, y2_index, y3_index, d,
                                            deviation, name)
    elif orientation >= math.pi:
        return outside_rectangle_deviation(box, orientation-math.pi,
                                           y1_index, y2_index, y3_index, d,
                                            deviation, name)
    position, length, width = box
    tmp = (position[0] * math.cos(orientation) +
           position[1] * math.sin(orientation) - length / 2,
           position[0] * math.cos(orientation) +
           position[1] * math.sin(orientation) + length / 2,
           -position[0] * math.sin(orientation) +
           position[1] * math.cos(orientation) - width / 2,
           -position[0] * math.sin(orientation) +
           position[1] * math.cos(orientation) + width / 2)
    y1_min, y1_max, y2_min, y2_max = tmp

    if orientation >= math.pi/2:
        return outside_rectangle_deviation((position, width, length), orientation-math.pi/2,
                                           y1_index, y2_index, y3_index, d,
                                            deviation, name)

    # Create predicates a*y >= b for each side of the rectangle
    a1 = np.zeros((1, d))
    a1[:, y1_index] = math.cos(orientation)
    a1[:, y2_index] = math.sin(orientation)
    right = line_angle_formula(a1, y1_max, y1_index, y2_index, y3_index, d, deviation=deviation)
    left = line_angle_formula(-a1, -y1_min, y1_index, y2_index, y3_index, d, deviation=deviation)

    a2 = np.zeros((1, d))
    a2[:, y1_index] = -math.sin(orientation)
    a2[:, y2_index] = math.cos(orientation)
    top = line_angle_formula(a2, y2_max, y1_index, y2_index, y3_index, d, deviation=deviation)
    bottom = line_angle_formula(-a2, -y2_min, y1_index, y2_index, y3_index, d, deviation=deviation)

    # Take the disjuction across all the sides
    outside_rectangle = right | left | top | bottom

    # set the names
    if name is not None:
        right.name = "right of " + name
        left.name = "left of " + name
        top.name = "top of " + name
        bottom.name = "bottom of " + name
        outside_rectangle.name = name
    outside_rectangle.simplify()
    return outside_rectangle

def inside_rectangle_deviation(box, orientation, y1_index, y2_index, y3_index, d,
                                     deviation=np.array([0, 0]), name=None): 
    ### a try for 4 vertices outside rectangle
    assert y1_index < d, "index must be less than signal dimension"
    assert y2_index < d, "index must be less than signal dimension"
    assert y3_index < d, "index must be less than signal dimension"
    
    if orientation < 0: 
        return inside_rectangle_deviation(box, orientation + math.pi,
                                          y1_index, y2_index, y3_index, d,
                                            deviation, name)
    elif orientation >= math.pi: 
        return inside_rectangle_deviation(box, orientation - math.pi,
                                          y1_index, y2_index, y3_index, d,
                                            deviation, name)

    position, length, width = box
    tmp = (position[0] * math.cos(orientation) +
           position[1] * math.sin(orientation) - length / 2,
           position[0] * math.cos(orientation) +
           position[1] * math.sin(orientation) + length / 2,
           -position[0] * math.sin(orientation) +
           position[1] * math.cos(orientation) - width / 2,
           -position[0] * math.sin(orientation) +
           position[1] * math.cos(orientation) + width / 2)
    y1_min, y1_max, y2_min, y2_max = tmp

    if orientation >= math.pi/2:
        return inside_rectangle_deviation((position, width, length), orientation - math.pi/2,
                                          y1_index, y2_index, y3_index, d,
                                            deviation, name)

    # Create predicates a*y >= b for each side of the rectangle
    a1 = np.zeros((1, d))
    a1[:, y1_index] = math.cos(orientation)
    a1[:, y2_index] = math.sin(orientation)
    right = line_angle_formula(-a1, -y1_max, y1_index, y2_index, y3_index, d, deviation=deviation)
    left = line_angle_formula(a1, y1_min, y1_index, y2_index, y3_index, d, deviation=deviation)

    a2 = np.zeros((1, d))
    a2[:, y1_index] = -math.sin(orientation)
    a2[:, y2_index] = math.cos(orientation)
    top = line_angle_formula(-a2, -y2_max, y1_index, y2_index, y3_index, d, deviation=deviation)
    bottom = line_angle_formula(a2, y2_min, y1_index, y2_index, y3_index, d, deviation=deviation)

    # Take the disjuction across all the sides
    inside_rectangle = right & left & top & bottom

    # set the names
    if name is not None:
        right.name = "right of " + name
        left.name = "left of " + name
        top.name = "top of " + name
        bottom.name = "bottom of " + name
        inside_rectangle.name = name
    inside_rectangle.simplify()
    return inside_rectangle

def inside_rectangle_formula(box,
                             orientation,
                             y1_index,
                             y2_index,
                             d,
                             deviation=np.array([0, 0]),
                             name=None):
    assert y1_index < d, "index must be less than signal dimension"
    assert y2_index < d, "index must be less than signal dimension"
    if orientation < 0:
        return inside_rectangle_formula(box,
                                        orientation + math.pi, y2_index,
                                        y1_index, d, deviation, name)
    elif orientation >= math.pi:
        return inside_rectangle_formula(box,
                                        orientation - math.pi, y2_index,
                                        y1_index, d, deviation, name)

    # Unpack the bounds
    position, length, width = box
    tmp = (position[0] * math.cos(orientation) +
           position[1] * math.sin(orientation) - length / 2,
           position[0] * math.cos(orientation) +
           position[1] * math.sin(orientation) + length / 2,
           -position[0] * math.sin(orientation) +
           position[1] * math.cos(orientation) - width / 2,
           -position[0] * math.sin(orientation) +
           position[1] * math.cos(orientation) + width / 2)
    y1_min, y1_max, y2_min, y2_max = tmp

    if orientation >= math.pi / 2:
        return inside_rectangle_formula((position, width, length),
                                        orientation - math.pi / 2, y2_index,
                                        y1_index, d, deviation, name)

    # Create predicates a*y >= b for each side of the rectangle
    a1 = np.zeros((1, d))
    a1[:, y1_index] = math.cos(orientation)
    a1[:, y2_index] = math.sin(orientation)
    right = LinearPredicate(
        a1,
        y1_min - a1[:, y1_index] * deviation[0] - a1[:, y2_index] * deviation[1])
    left = LinearPredicate(
        -a1, -y1_max + a1[:, y1_index] * deviation[0] +
        a1[:, y2_index] * deviation[1])

    a2 = np.zeros((1, d))
    a2[:, y1_index] = -math.sin(orientation)
    a2[:, y2_index] = math.cos(orientation)
    bottom = LinearPredicate(
        a2,
        y2_min - a2[:, y1_index] * deviation[0] - a2[:, y2_index] * deviation[1])
    top = LinearPredicate(
        -a2, -y2_max + a2[:, y1_index] * deviation[0] +
        a2[:, y2_index] * deviation[1])

    # Take the conjuction across all the sides
    inside_rectangle = right & left & top & bottom

    # set the names
    if name is not None:
        right.name = "right of " + name
        left.name = "left of " + name
        top.name = "top of " + name
        bottom.name = "bottom of " + name
        inside_rectangle.name = name
    inside_rectangle.simplify()
    return inside_rectangle

def inside_polygon_formula(vertices, ###### n*2 matrix, inside right area
                             y1_index,
                             y2_index,
                             d,
                             name=None):
    assert y1_index < d, "index must be less than signal dimension"
    assert y2_index < d, "index must be less than signal dimension"
    n = vertices.shape[0]
    Predicate_list = []
    for i in range(n):
        [x1, y1] = vertices[i-1, :]
        [x2, y2] = vertices[i, :]
        if y2==y1:
            assert x2!=x1
            a = np.zeros((1, d))
            a[:, y1_index] = 0
            a[:, y2_index] = -(x2-x1)/abs(x2-x1)
            Predicate_list.append(LinearPredicate(a, -y1))
            continue
        v = np.array([x2-x1,y2-y1])
        v = v/np.linalg.norm(v)
        a = np.zeros((1, d))
        a[:, y1_index] = v[1]
        a[:, y2_index] = -v[0]
        Predicate_list.append(LinearPredicate(a, v[1]*x1-v[0]*y1))
    result = Predicate_list[0]
    for i in range(1, n):
        result = result & Predicate_list[i]
    result.simplify()
    return result

def inside_polygon_deviation_formula(vertices, ###### n*2 matrix, inside right area
                             y1_index,
                             y2_index,
                             y3_index,
                             d, b=0,
                             name=None):
    assert y1_index < d, "index must be less than signal dimension"
    assert y2_index < d, "index must be less than signal dimension"
    n = vertices.shape[0]
    assert n==4, ""
    Predicate_list = []
    for i in range(n):
        [x1, y1] = vertices[i-1, :]
        [x2, y2] = vertices[i, :]
        if y2==y1:
            assert x2!=x1
            a = np.zeros((1, d))
            a[:, y1_index] = 0
            a[:, y2_index] = -(x2-x1)/abs(x2-x1)
            Predicate_list.append(line_angle_formula(a, -y1, y1_index, y2_index, y3_index, d, [b, 0]))
        elif x2==x1:
            a = np.zeros((1, d))
            a[:, y1_index] = (y2-y1)/abs(y2-y1)
            a[:, y2_index] = 0
            Predicate_list.append(line_angle_formula(a, x1, y1_index, y2_index, y3_index, d, [b, 0]))
        else:
            v = np.array([x2-x1,y2-y1])
            v = v/np.linalg.norm(v)
            a = np.zeros((1, d))
            a[:, y1_index] = v[1]
            a[:, y2_index] = -v[0]
            Predicate_list.append(line_angle_formula(a, v[1]*x1-v[0]*y1,
                                                     y1_index, y2_index, y3_index, d, [b, 0]))
        # elif y2<y1:
        #     v = np.array([x2-x1,y2-y1])
        #     v = v/np.linalg.norm(v)
        #     a = np.zeros((1, d))
        #     a[:, y1_index] = -v[1]
        #     a[:, y2_index] = v[0]
        #     Predicate_list.append(line_angle_formula(a, -v[1]*x1+v[0]*y1,
        #                                              y1_index, y2_index, y3_index, d, [b, 0]))
    result = Predicate_list[0]
    for i in range(1, n):
        result = result & Predicate_list[i]
    result.simplify()
    return result

def outside_rectangle_formula(box,
                              orientation,
                              y1_index,
                              y2_index,
                              d,
                              deviation=np.array([0, 0]),
                              name=None):
    assert y1_index < d, "index must be less than signal dimension"
    assert y2_index < d, "index must be less than signal dimension"
    if orientation < 0:
        return outside_rectangle_formula(box,
                                         orientation + math.pi, y2_index,
                                         y1_index, d, deviation, name)
    elif orientation >= math.pi:
        return outside_rectangle_formula(box,
                                         orientation - math.pi, y2_index,
                                         y1_index, d, deviation, name)

    # Unpack the bounds
    position, length, width = box
    tmp = (position[0] * math.cos(orientation) +
           position[1] * math.sin(orientation) - length / 2,
           position[0] * math.cos(orientation) +
           position[1] * math.sin(orientation) + length / 2,
           -position[0] * math.sin(orientation) +
           position[1] * math.cos(orientation) - width / 2,
           -position[0] * math.sin(orientation) +
           position[1] * math.cos(orientation) + width / 2)
    y1_min, y1_max, y2_min, y2_max = tmp

    if orientation >= math.pi / 2:
        return outside_rectangle_formula((position, width, length),
                                         orientation - math.pi / 2, y2_index,
                                         y1_index, d, deviation, name)

    # Create predicates a*y >= b for each side of the rectangle
    a1 = np.zeros((1, d))
    a1[:, y1_index] = math.cos(orientation)
    a1[:, y2_index] = math.sin(orientation)
    right = LinearPredicate(
        a1,
        y1_max - a1[:, y1_index] * deviation[0] - a1[:, y2_index] * deviation[1])
    left = LinearPredicate(
        -a1, -y1_min + a1[:, y1_index] * deviation[0] +
        a1[:, y2_index] * deviation[1])

    a2 = np.zeros((1, d))
    a2[:, y1_index] = -math.sin(orientation)
    a2[:, y2_index] = math.cos(orientation)
    top = LinearPredicate(
        a2,
        y2_max - a2[:, y1_index] * deviation[0] - a2[:, y2_index] * deviation[1])
    bottom = LinearPredicate(
        -a2, -y2_min + a2[:, y1_index] * deviation[0] +
        a2[:, y2_index] * deviation[1])

    # Take the disjuction across all the sides
    outside_rectangle = right | left | top | bottom

    # set the names
    if name is not None:
        right.name = "right of " + name
        left.name = "left of " + name
        top.name = "top of " + name
        bottom.name = "bottom of " + name
        outside_rectangle.name = name

    return outside_rectangle


def upon_poly_formula(poly,
                      y1_index,
                      y2_index,
                      d,
                      deviation=np.array([0, 0]),
                      name=None):
    # Define the predicate function g(y) >= 0
    def g(y):
        y1 = y[y1_index] + deviation[0]
        y2 = y[y2_index] + deviation[1]
        t = 0
        tmp = 1
        for i in range(len(poly)):
            t += poly[len(poly) - 1 - i] * tmp
            tmp *= y1

        return y2 - t

    return NonlinearPredicate(g, d, name=name)


def front_formula(center_position, orientation,
                         y_index_list, center_index_list,
                          d, name=None):
    def g(y):
        x0 = y[y_index_list[0]]
        y0 = y[y_index_list[1]]
        angle = y[y_index_list[2]]
        centerx = y[center_index_list[0]]
        centery = y[center_index_list[1]]
        center_angle = y[center_index_list[2]]
        if isinstance(y, np.ndarray):
            x = centerx - x0
            y = centery - y0
            return x*np.cos(angle)+y*np.sin(angle)
        else:
            x = centerx - x0
            y = centery - y0
            return x*sp.cos(angle)+y*sp.sin(angle)
    return MovingNonlinearPredicate(g, d, np.vstack((center_position, orientation)), name=name)

def getBoundarySTL(lanelets):
    for i, lanelet in enumerate(lanelets):
        if i==0:
            boundary_right = lanelets[lanelet].right_vertices
        elif i==len(lanelets)-1:
            boundary_left = lanelets[lanelet].left_vertices
    z_right = np.polyfit(boundary_right[:, 0], boundary_right[:, 1], 3)
    z_left = np.polyfit(boundary_left[:, 0], boundary_left[:, 1], 3)
    right_formula = upon_poly_formula(z_right, 0, 1, 5, deviation=[0, -2*Width/3])
    lef_formula = upon_poly_formula(z_left, 0, 1, 5, deviation=[0, 2*Width/3])
    lef_formula = lef_formula.negation()
    road_stl = right_formula & lef_formula

    return road_stl, z_right, boundary_right, z_left, boundary_left


def getStaticObstaclesSTL(obstacles: Dict[int, StaticObstacle], dynamic_model):
    if len(obstacles)==0:
        return None
    obstacle_formulas = []
    for i, obstacle in enumerate(obstacles):
        position = obstacle.initial_state.position
        length = obstacle.obstacle_shape.length
        width = obstacle.obstacle_shape.width
        orientation = obstacle.initial_state.orientation
        obs = (position, length, width)
        obstacle_formulas.append(
            two_rectangle_outside_formula(
            obs, orientation, 0, 1, 4, 5, dynamic_model.parameters.b))
    obstacle_avoidance = obstacle_formulas[0]
    for i in range(1, len(obstacle_formulas)):
        obstacle_avoidance = obstacle_avoidance & obstacle_formulas[i]
    if len(obstacle_formulas) > 1:
        obstacle_avoidance.simplify()
    return obstacle_avoidance


def getInitialState(planning_problem_dict: PlanningProblemSet, dynamic_model):
    ####TODO what does the number of planning problem means
    for i, problem in enumerate(planning_problem_dict):
        initial_state = planning_problem_dict[problem].initial_state
    initial_state_new = dynamic_model._state_to_array(initial_state)
    return initial_state_new[0]


def getFinalState(planning_problem_dict: PlanningProblemSet, dynamic_model):
    ####TODO what does the number of planning problem means
    for i, problem in enumerate(planning_problem_dict):
        initial_state = planning_problem_dict[problem].initial_state
        goal = planning_problem_dict[problem]._goal_region
        K = goal.state_list[0].time_step.end + 1
        final_state = copy.deepcopy(initial_state)
        final_state.position = goal.state_list[0].position.center
        final_state.orientation = goal.state_list[0].position.orientation
    final_state_new = dynamic_model._state_to_array(final_state)
    return K, final_state_new[0]

def getFinalStateLimit(planning_problem_dict: PlanningProblemSet, dynamic_model):
    ####TODO what does the number of planning problem means
    for i, problem in enumerate(planning_problem_dict):
        initial_state = planning_problem_dict[problem].initial_state
        goal = planning_problem_dict[problem]._goal_region
        K = goal.state_list[0].time_step.length
        final_state = copy.deepcopy(initial_state)
        final_state.position = goal.state_list[0].position.center
        final_state.orientation = goal.state_list[0].position.orientation
        width = goal.state_list[0].position.width
        length = goal.state_list[0].position.length
    final_state_new = dynamic_model._state_to_array(final_state)
    final_position_max = np.array([final_state_new[0][0] + length/2, final_state_new[0][1] + width/2])
    final_position_min = np.array([final_state_new[0][0] - length/2, final_state_new[0][1] - width/2])
    return final_position_max, final_position_min


def getGoalSTL(planning_problem_dict: PlanningProblemSet, dynamic_model):
    ####TODO what does the number of planning problem means
    if len(planning_problem_dict) == 0:
        return None
    goal_formulas = []
    for i, problem in enumerate(planning_problem_dict):
        initial_state = planning_problem_dict[problem].initial_state
        goal = planning_problem_dict[problem]._goal_region
        K = goal.state_list[0].time_step.length
        position = goal.state_list[0].position.center
        orientation = goal.state_list[
            0].position.orientation  ###TODO can use vertices
        length = goal.state_list[0].position.length
        width = goal.state_list[0].position.width
        goal_area = (position, length, width)
        goal_formulas.append(
            inside_rectangle_deviation(
            goal_area, orientation, 0, 1, 4, 5, 
            [dynamic_model.parameters.b, 0]))
        
    goal_stl = goal_formulas[0]
    for i in range(1, len(goal_formulas)):
        goal_stl = goal_stl | goal_formulas[i]
    if len(goal_formulas) > 1:
        goal_stl.simplify()
    return goal_stl

def getDynamicObstaclesSTL(obstacles: Dict[int, DynamicObstacle], K, dynamic_model):
    if len(obstacles)==0:
        return None
    obstacle_formulas = []
    for i, obstacle in enumerate(obstacles):
        position = np.zeros((2, K))
        orientation = np.zeros((1, K))
        for k in range(K):
            if k == 0 :
                position[:, 0] = obstacle.initial_state.position
                orientation[0, 0] = obstacle.initial_state.orientation
            position[:, k] = obstacle.prediction.occupancy_set[k - 1].shape.center
            orientation[0, k] = obstacle.prediction.occupancy_set[k - 1].shape.orientation
        length = obstacle.obstacle_shape.length
        width = obstacle.obstacle_shape.width
        obs = (position, length, width)
        obstacle_formulas.append(
            two_moving_rectangle_outside_formula(obs, orientation,
                                                 [0, 1, 4], [5, 6, 7], 5,
                                                 dynamic_model.parameters.b))
    obstacle_avoidance = obstacle_formulas[0]
    for i in range(1, len(obstacle_formulas)):
        obstacle_avoidance = obstacle_avoidance & obstacle_formulas[i]
    if len(obstacle_formulas) > 1:
        obstacle_avoidance.simplify()
    return obstacle_avoidance

def getInSameLaneAndFrontSTL(init_state, obstacles: Dict[int, DynamicObstacle], K,
                             left_lane, right_lane, dynamic_model): 
    obstacle_formulas = []
    for i, obstacle in enumerate(obstacles):
        position = np.zeros((2, K))
        orientation = np.zeros((1, K))
        for k in range(K):
            if k == 0 :
                position[:, 0] = obstacle.initial_state.position
                orientation[0, 0] = obstacle.initial_state.orientation
            position[:, k] = obstacle.prediction.occupancy_set[k - 1].shape.center
            orientation[0, k] = obstacle.prediction.occupancy_set[k - 1].shape.orientation
        length = obstacle.obstacle_shape.length
        width = obstacle.obstacle_shape.width
        obs = (position, length, width)
        is_init_front = ((position[0, 0] - init_state[0])*np.cos(init_state[4]) +\
              (position[1, 0] - init_state[1])*np.sin(init_state[4])>0)
        is_below_left_lane = (position[1, 0] - np.polyval(left_lane, position[0, 0])<0)
        is_above_right_lane = (position[1, 0] - np.polyval(right_lane, position[0, 0])>0)
        if is_init_front & is_below_left_lane & is_above_right_lane:
            obstacle_formulas.append(two_moving_rectangle_outside_formula(obs, orientation,
                                                        [0, 1, 4], [5, 6, 7], 5,
                                                        dynamic_model.parameters.b,
                                                        4.0))
    
    obstacle_avoidance = obstacle_formulas[0]
    for i in range(1, len(obstacle_formulas)):
        obstacle_avoidance = obstacle_avoidance & obstacle_formulas[i]
    if len(obstacle_formulas) > 1:
        obstacle_avoidance.simplify()
    return obstacle_avoidance


def getCenterTrajectory(X, dynamic_model):
    X_center = np.zeros((2, X.shape[1]))
    X_center[0, :] = X[0, :] + dynamic_model.parameters.b*np.cos(X[4, :])
    X_center[1, :] = X[1, :] + dynamic_model.parameters.b*np.sin(X[4, :])
    return X_center

def getProblemID(planning_problem_dict: PlanningProblemSet):
    ####TODO what does the number of planning problem means
    for i, problem in enumerate(planning_problem_dict):
        id = planning_problem_dict[problem].planning_problem_id
        break
    return id


def checkTrajectory(X_converge, dynamic_model, planning_problem_set, scenario):
    state_list = []

    for i in range(X_converge.shape[1]):
        state_list.append(dynamic_model._array_to_state(X_converge[:, i], i))
    trajectory_solution = Trajectory(0, state_list)
    # from SMP.motion_planner.utility import visualize_solution
    # visualize_solution(scenario, planning_problem_set, trajectory_solution)
    # create PlanningProblemSolution object
    kwarg = {'planning_problem_id': getProblemID(planning_problem_set._planning_problem_dict),
            'vehicle_model':VehicleModel.KS,                            # used vehicle model, change if needed
            'vehicle_type':VehicleType.BMW_320i,                        # used vehicle type, change if needed
            'cost_function':CostFunction.SA1,                           # cost funtion, DO NOT use JB1
            'trajectory':trajectory_solution}

    planning_problem_solution = PlanningProblemSolution(**kwarg)

    # create Solution object
    kwarg = {'scenario_id':scenario.scenario_id,
            'planning_problem_solutions':[planning_problem_solution]}

    solution = Solution(**kwarg)
    result = valid_solution(scenario, planning_problem_set, solution)
    return result, solution



    

