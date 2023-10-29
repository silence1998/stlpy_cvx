from stlpy.systems.linear import LinearSystem
import numpy as np

class MyDoubleIntegrator(LinearSystem):
    """
    A linear system describing a double integrator in :math:`d` dimensions
    with full state and control output:
    """
    def __init__(self, d, dt):
        I = np.eye(d)
        z = np.zeros((d,d))

        A = np.block([[I,I * dt],
                      [z,I]])
        B = np.block([[z],
                      [I * dt]])
        C = np.block([[I,z],
                      [z,I]])
        D = np.block([[z],
                      [z]])

        LinearSystem.__init__(self, A, B, C, D)