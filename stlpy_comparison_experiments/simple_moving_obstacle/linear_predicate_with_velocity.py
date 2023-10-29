from stlpy.benchmarks.common import (inside_rectangle_formula,
                     outside_rectangle_formula,
                     make_rectangle_patch)

from stlpy.STL.formula import STLFormula

import numpy as np

class LinearPredicateWithVelocity(STLFormula):
    """
    A linear STL predicate :math:`\pi` defined by

    .. math::

        a^Ty_t - b - v*t \geq 0

    where :math:`y_t \in \mathbb{R}^d` is the value of the signal
    at a given timestep :math:`t`, :math:`a \in \mathbb{R}^d`,
    and :math:`b \in \mathbb{R}`.

    :param a:       a numpy array or list representing the vector :math:`a`
    :param b:       a list, numpy array, or scalar representing :math:`b`
    :param name:    (optional) a string used to identify this predicate.
    """
    def __init__(self, a, b, v, name=None):
        # Convert provided constraints to numpy arrays
        self.a = np.asarray(a).reshape((-1,1))
        self.b = np.atleast_1d(b)
        self.v = np.atleast_1d(v)
        # Some dimension-related sanity checks
        assert (self.a.shape[1] == 1), "a must be of shape (d,1)"
        assert (self.b.shape == (1,)), "b must be of shape (1,)"
        assert (self.v.shape == (1,)), "v must be of shape (1,)"

        # Store the dimensionality of y_t
        self.d = self.a.shape[0]

        # A unique string describing this predicate
        self.name = name

    def negation(self):
        if self.name is None:
            newname = None
        else:
            newname = "not " + self.name
        return LinearPredicateWithVelocity(-self.a, -self.b, -self.v, name=newname)

    def robustness(self, y, t):
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert isinstance(t, int), "timestep t must be an integer"
        assert y.shape[0] == self.d, "y must be of shape (d,T)"
        assert y.shape[1] > t, "requested timestep %s, but y only has %s timesteps" % (t, y.shape[1])

        return self.a.T@y[:,t] - self.b - self.v * t

    def is_predicate(self):
        return True

    def is_state_formula(self):
        return True

    def is_disjunctive_state_formula(self):
        return True

    def is_conjunctive_state_formula(self):
        return True

    def get_all_inequalities(self):
        ###!!!!!!!!!!!!! TODO without t, can't establish the inequation
        A = -self.a.T
        b = -self.b
        return (A,b)

    def __str__(self):
        if self.name is None:
            return "{ Predicate %s*y >= %s }" % (self.a, self.b)
        else:
            return "{ Predicate " + self.name + " }"

def outside_rectangle_formula_with_velocity(bounds, y1_index, y2_index, v, d, name=None):
    """
    Create an STL formula representing being outside a
    rectangle with the given bounds:

    ::

       y2_max   +-------------------+
                |                   |
                |                   |
                |                   |
       y2_min   +-------------------+
                y1_min              y1_max

    :param bounds:      Tuple ``(y1_min, y1_max, y2_min, y2_max)`` containing
                        the bounds of the rectangle. 
    :param y1_index:    index of the first (``y1``) dimension
    :param y2_index:    index of the second (``y2``) dimension
    :param d:           dimension of the overall signal
    :param name:        (optional) string describing this formula
    
    :return outside_rectangle:   An ``STLFormula`` specifying being outside the
                                 rectangle at time zero.
    """
    assert y1_index < d , "index must be less than signal dimension"
    assert y2_index < d , "index must be less than signal dimension"

    # Unpack the bounds
    y1_min, y1_max, y2_min, y2_max = bounds

    # Create predicates a*y >= b for each side of the rectangle
    a1 = np.zeros((1,d)); a1[:,y1_index] = 1
    right = LinearPredicateWithVelocity(a1, y1_max, v[0])
    left = LinearPredicateWithVelocity(-a1, -y1_min, -v[0])

    a2 = np.zeros((1,d)); a2[:,y2_index] = 1
    top = LinearPredicateWithVelocity(a2, y2_max, v[1])
    bottom = LinearPredicateWithVelocity(-a2, -y2_min, -v[1])

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

def inside_rectangle_formula_with_velocity(bounds, y1_index, y2_index, v, d, name=None):
    """
    Create an STL formula representing being outside a
    rectangle with the given bounds:

    ::

       y2_max   +-------------------+
                |                   |
                |                   |
                |                   |
       y2_min   +-------------------+
                y1_min              y1_max

    :param bounds:      Tuple ``(y1_min, y1_max, y2_min, y2_max)`` containing
                        the bounds of the rectangle. 
    :param y1_index:    index of the first (``y1``) dimension
    :param y2_index:    index of the second (``y2``) dimension
    :param d:           dimension of the overall signal
    :param name:        (optional) string describing this formula
    
    :return outside_rectangle:   An ``STLFormula`` specifying being outside the
                                 rectangle at time zero.
    """
    assert y1_index < d , "index must be less than signal dimension"
    assert y2_index < d , "index must be less than signal dimension"

    # Unpack the bounds
    y1_min, y1_max, y2_min, y2_max = bounds

    # Create predicates a*y >= b for each side of the rectangle
    a1 = np.zeros((1,d)); a1[:,y1_index] = 1
    right = LinearPredicateWithVelocity(a1, y1_min, v[0])
    left = LinearPredicateWithVelocity(-a1, -y1_max, -v[0])

    a2 = np.zeros((1,d)); a2[:,y2_index] = 1
    top = LinearPredicateWithVelocity(a2, y2_min, v[1])
    bottom = LinearPredicateWithVelocity(-a2, -y2_max, -v[1])

    # Take the disjuction across all the sides
    outside_rectangle = right & left & top & bottom

    # set the names
    if name is not None:
        right.name = "right of " + name
        left.name = "left of " + name
        top.name = "top of " + name
        bottom.name = "bottom of " + name
        outside_rectangle.name = name

    return outside_rectangle

