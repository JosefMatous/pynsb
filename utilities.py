import numpy as np
import time
from scipy.linalg import block_diag
from scipy.integrate import solve_ivp

import imcpy

def rotation_matrix(pitch, yaw):
    c_theta = np.cos(pitch)
    s_theta = np.sin(pitch)
    c_psi = np.cos(yaw)
    s_psi = np.sin(yaw)

    R = np.array([[c_psi*c_theta, -s_psi, c_psi*s_theta],
                [s_psi*c_theta, c_psi, s_psi*s_theta],
                [-s_theta, 0., c_theta]])
    return R

def rotation_matrix_derivative(pitch, yaw, pitch_rate, yaw_rate):
    p = - yaw_rate * np.sin(pitch)
    q = pitch_rate
    r = yaw_rate * np.cos(pitch)

    S = np.array([[ 0, -r,  q],
                [ r,  0, -p],
                [-q,  p,  0]])
    R = rotation_matrix(pitch, yaw)

    return R.dot(S)
    
def calculate_barycenter(positions):
    n = len(positions)

    p_b = np.zeros(n)
    for p in positions:
        p_b += p / n

    return p_b

def predict_vehicle_state(estate: imcpy.EstimatedState):
    """
    Predicts the current state of the vehicle based on a previous estimate

    Updates the given state with a prediction of the vehicle's current state.
    The function assumes that the vehicle's velocities have not changed.
    The prediction is thus found by solving the kinematic ODEs of the vehicle.
    """
    eta0 = np.array([estate.x, estate.y, estate.z, estate.phi, estate.theta, estate.psi])
    nu = np.array([estate.u, estate.v, estate.w, estate.p, estate.q, estate.r])
    ode = lambda _, x: transformation_matrix(x).dot(nu)
    my_timestamp = time.time()
    sol = solve_ivp(ode, [estate.timestamp, my_timestamp], eta0, t_eval=[my_timestamp])
    etaT = sol.y[:, -1]

    estate.x = etaT[0]
    estate.y = etaT[0]
    estate.z = etaT[0]
    estate.phi = etaT[0]
    estate.theta = etaT[0]
    estate.psi = etaT[0]

    estate.set_timestamp_now()


def transformation_matrix(eta):
    phi, theta, psi = eta[3], eta[3], eta[5]
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth  = np.cos(theta)
    sth  = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    # Rotation matrix
    R = np.array([[cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth],
                [spsi*cth, cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi],
                [-sth, cth*sphi, cth*cphi ]])
    # Angular rate matrix
    T = np.array([[1., sphi*sth/cth, cphi*sth/cth],
                  [0., cphi, -sphi],
                  [0., sphi/cth, cphi/cth]])
    
    return block_diag(R, T)
