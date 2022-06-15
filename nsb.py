from los import LineOfSight
from formation import FormationKeeping
from geompath import GeometricPath
from utilities import calculate_barycenter

import numpy as np

import imcpy

def step(positions, path_parameter, path: GeometricPath, los: LineOfSight, form: FormationKeeping):
    path_ref = path.getPathReference(path_parameter)
    vel_los, param_derivative = los.step(path_ref, calculate_barycenter(positions))
    vel_form = form.get_velocities(path_ref, positions, param_derivative)

    vel_nsb = [vel_los + v_f_i for v_f_i in vel_form]

    return vel_nsb, param_derivative

def follow_the_carrot_reference(estates, path_parameter, path: GeometricPath, los: LineOfSight, form: FormationKeeping, T_carrot=15.):
    positions = [np.array([e.x, e.y, e.z]) for e in estates]
    vel_nsb, _ = step(positions, path_parameter, path, los, form)

    refs = []
    for i in range(len(vel_nsb)):
        v_i = vel_nsb[i]
        U = np.linalg.norm(v_i)
        p_i = positions[i]
        p_ref_i = p_i + T_carrot * v_i

        lat, lon = imcpy.coordinates.WGS84.displace(estates[i].lat, estates[i].lon, n=p_ref_i[0], e=p_ref_i[1])
        r = imcpy.Reference()
        r.lat = lat  # Target waypoint
        r.lon = lon  # Target waypoint

        # Assign z
        dz = imcpy.DesiredZ()
        dz.value = p_ref_i[2]
        dz.z_units = imcpy.ZUnits.DEPTH
        r.z = dz

        # Assign the speed
        ds = imcpy.DesiredSpeed()
        ds.value = U
        ds.speed_units = imcpy.SpeedUnits.METERS_PS
        r.speed = ds

        # Bitwise flags (see IMC spec for explanation)
        flags = imcpy.Reference.FlagsBits.LOCATION | imcpy.Reference.FlagsBits.SPEED | imcpy.Reference.FlagsBits.Z
        r.flags = flags

        refs.append(r)
    
    return refs

def simulated_response_reference(estates, path_parameter, path: GeometricPath, los: LineOfSight, form: FormationKeeping,
                                 T_sim=2.5, L_carrot=25.):    
    n = len(estates)
    pos0 = [np.array([e.x, e.y, e.z]) for e in estates]
    pos = [p.copy() for p in pos0]
    U = np.zeros(n)
    _nsb_forward_euler(pos, path_parameter, U, path, los, form, T_sim, int(T_sim*10))
    U /= T_sim

    refs = []
    for i in range(n):
        v_i = pos[i] - pos0[i]
        v_n = v_i / np.linalg.norm(v_i)
        p_ref_i = pos[i] + L_carrot * v_n

        lat, lon = imcpy.coordinates.WGS84.displace(estates[i].lat, estates[i].lon, n=p_ref_i[0], e=p_ref_i[1])
        r = imcpy.Reference()
        r.lat = lat  # Target waypoint
        r.lon = lon  # Target waypoint

        # Assign z
        dz = imcpy.DesiredZ()
        dz.value = p_ref_i[2]
        dz.z_units = imcpy.ZUnits.DEPTH
        r.z = dz

        # Assign the speed
        ds = imcpy.DesiredSpeed()
        ds.value = U[i]
        ds.speed_units = imcpy.SpeedUnits.METERS_PS
        r.speed = ds

        # Bitwise flags (see IMC spec for explanation)
        flags = imcpy.Reference.FlagsBits.LOCATION | imcpy.Reference.FlagsBits.SPEED | imcpy.Reference.FlagsBits.Z
        r.flags = flags

        refs.append(r)
    
    return refs

def _nsb_forward_euler(positions, path_parameter, Ui, path: GeometricPath, los: LineOfSight, form: FormationKeeping, T, n_steps):
    param_copy = path_parameter
    for i in range(n_steps):
        param_copy = _nsb_forward_euler_step(positions, param_copy, Ui, path, los, form, T / n_steps)

def _nsb_forward_euler_step(positions, path_parameter, Ui, path: GeometricPath, los: LineOfSight, form: FormationKeeping, dt):
    vel_nsb, param_derivative = step(positions, path_parameter, path, los, form)
    for i in range(len(positions)):
        positions[i] += vel_nsb[i] * dt
        Ui[i] += np.linalg.norm(vel_nsb[i]) * dt
    return path_parameter + param_derivative * dt
