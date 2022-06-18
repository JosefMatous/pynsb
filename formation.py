import utilities

import numpy as np

class FormationKeeping:
    def __init__(self, shape=None, gain=0.05):
        if shape is None:
            self.shape = [np.array([6.,0.,0.]),
                        np.array([-3.,15.,0.]),
                        np.array([3.,-15.,0.])]
        else:
            self.shape = shape

        self.gain = gain

    def step(self, path_ref, positions, path_parameter_derivative):
        n = len(positions)
        assert(n <= len(self.shape))

        barycenter = utilities.calculate_barycenter(positions)

        R = utilities.rotation_matrix(path_ref.angles[0], path_ref.angles[1])
        R_dot = utilities.rotation_matrix_derivative(path_ref.angles[0], path_ref.angles[1], 
            path_ref.curvature[0] * path_parameter_derivative, path_ref.curvature[1] * path_parameter_derivative)
        
        rotated_shape = [R.dot(x) for x in self.shape]
        z_ref = [path_ref.position[2] + s[2] for s in rotated_shape]
        rotated_shape_derivative = [R_dot.dot(x) for x in self.shape]

        form_error = [positions[i] - barycenter - rotated_shape[i] for i in range(n)]
        velocities = [rotated_shape_derivative[i] - self.gain * form_error[i] for i in range(n)]
        return velocities, z_ref

    