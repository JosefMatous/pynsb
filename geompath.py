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

    @staticmethod
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

class OscillatingEllipse(GeometricPath):
    def __init__(self, center=None, a=100., b=50., yaw=0., phi0=0., clockwise=True, z_amplitude=1., z_frequency=2., z_phi0=np.pi/2):
        if center is None:
            self.center = np.zeros(3)
        else:
            self.center = center
        self.a = a
        self.b = b
        self.yaw = yaw
        self.phi0 = phi0
        self.clockwise = clockwise
        self.z_amplitude = z_amplitude
        self.z_frequency = z_frequency
        self.z_phi0 = z_phi0

    def getPathReference(self, path_parameter):
        R_yaw = utilities.rotation_matrix(0., self.yaw)

        direction = 1. if self.clockwise else -1.
        arg = self.phi0 + direction * path_parameter
        s_arg = np.sin(arg)
        c_arg = np.cos(arg)
        z_arg = self.z_phi0 + self.z_frequency*path_parameter
        s_z_arg = np.sin(z_arg)
        c_z_arg = np.cos(z_arg)
        pn = np.array([self.a*c_arg, self.b*s_arg, self.z_amplitude*s_z_arg])
        p = self.center + R_yaw.dot(pn)

        xn_diff = -direction*self.a*s_arg
        yn_diff = direction*self.b*c_arg
        zn_diff = self.z_amplitude*self.z_frequency*c_z_arg
        p_diff = R_yaw.dot(np.array([xn_diff, yn_diff, zn_diff]))
        D2 = xn_diff**2 + yn_diff**2 + zn_diff**2
        path_gradient = np.sqrt(D2)
        #path_angles = np.array([-np.arcsin(zn_diff / path_gradient), self.yaw + np.arctan2(yn_diff, xn_diff)])        
        path_angles = np.array([-np.arcsin(p_diff[2] / path_gradient), np.arctan2(p_diff[1], p_diff[0])])        

        xn_ddiff = -self.a*c_arg
        yn_ddiff = -self.b*s_arg
        zn_ddiff = -self.z_amplitude*(self.z_frequency**2)*s_z_arg
        d2 = xn_diff**2 + yn_diff**2
        d = np.sqrt(d2)
        kappa = (xn_ddiff*xn_diff*zn_diff + yn_ddiff*yn_diff*zn_diff)/(d*D2) - zn_ddiff*d/D2
        iota = (yn_ddiff*xn_diff - xn_ddiff*yn_diff) / d2
        path_curvature = np.array([kappa, iota])

        return PathReference(p, path_angles, path_curvature, path_gradient)
