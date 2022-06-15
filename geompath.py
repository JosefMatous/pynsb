"""
Geometric paths
"""

import numpy as np

import utilities

class PathReference:
    def __init__(self, position, angles, curvature, gradient):
        self.position = position
        self.angles = angles
        self.curvature = curvature
        self.gradient = gradient

class GeometricPath:
    def __init__(self):
        raise NotImplementedError('GeometricPath is abstract')

    def getPathReference(self, path_parameter):
        raise NotImplementedError('Must be implemented in derived class')

    def getPathFollowingError(path_ref: PathReference, position):
        pos_err = position - path_ref.position # position error in inertial coordinates

        R = utilities.rotation_matrix(path_ref.angles[0], path_ref.angles[1])
        return R.transpose().dot(pos_err) # transformation to path-tangential coordinates

class PlanarSineWave(GeometricPath):
    def __init__(self, p0=None, amplitude=15., wavenumber=0.0419, yaw=0.):
        if p0 is None:
            self.p0 = np.zeros(3)
        else:
            self.p0 = p0
        self.amplitude = amplitude
        self.wavenumber = wavenumber
        self.yaw = yaw

    def getPathReference(self, path_parameter):
        R_yaw = utilities.rotation_matrix(0., self.yaw)

        arg = self.wavenumber * path_parameter
        pn = np.array([path_parameter, self.amplitude * np.sin(arg), 0.])
        p = self.p0 + R_yaw.dot(pn)

        yn_diff = self.amplitude * self.wavenumber * np.cos(arg)
        path_angles = np.array([0., self.yaw + np.arctan(yn_diff)])
        path_gradient = np.sqrt(1 + yn_diff**2)

        yn_diff2 = - (self.wavenumber**2) * pn[1]
        path_curvature = np.array([0., yn_diff2 / (yn_diff**2 + 1.)])

        return PathReference(p, path_angles, path_curvature, path_gradient)

