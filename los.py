import utilities
import geompath

import numpy as np

class LineOfSight:
    def __init__(self, lookahead=15., adaptive=False, speed=1., gain=0.2):
        self.lookahead = lookahead
        self.adaptive = adaptive
        self.speed = speed
        self.gain = gain

    def step(self, path_ref, position):
        path_err = geompath.GeometricPath.getPathFollowingError(path_ref, position)

        lookahead = self.lookahead
        if self.adaptive:
            lookahead = np.sqrt(self.lookahead**2 + path_err[0]**2 + path_err[1]**2 + path_err[2]**2)

        D = np.sqrt(lookahead**2 + path_err[1]**2 + path_err[2]**2)
        pitch_D = np.arcsin(path_err[2] / D)
        yaw_D = - np.arctan2(path_err[1], lookahead)
        R_d = utilities.rotation_matrix(pitch_D, yaw_D)

        R_path = utilities.rotation_matrix(path_ref.angles[0], path_ref.angles[1])

        velocity = R_d.dot(R_path.dot(np.array([self.speed, 0., 0.])))

        path_parameter_derivative = self.speed * (lookahead / D + self.gain * path_err[0] / np.sqrt(path_err[0]**2 + 1.))

        return velocity, path_parameter_derivative
