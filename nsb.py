from los import LineOfSight
from formation import FormationKeeping
from geompath import GeometricPath
from utilities import calculate_barycenter

import numpy as np

import imcpy

class NSBAlgorithm:
    def __init__(self, path: GeometricPath, los: LineOfSight, form: FormationKeeping, lat_home, lon_home):
        self.path = path
        self.los = los
        self.form = form
        self.lat_home = lat_home
        self.lon_home = lon_home

    def step(self, positions, path_parameter):
        path_ref = self.path.getPathReference(path_parameter)
        vel_los, param_derivative = self.los.step(path_ref, calculate_barycenter(positions))
        vel_form = self.form.get_velocities(path_ref, positions, param_derivative)

        vel_nsb = [vel_los + v_f_i for v_f_i in vel_form]

        return vel_nsb, param_derivative

    def _get_relative_position(self, estate: imcpy.EstimatedState):
        lat_v, lon_v = imcpy.coordinates.WGS84.displace(estate.lat, estate.lon, n=estate.x, e=estate.y)
        x_v, y_v, _ = imcpy.coordinates.WGS84.displacement(self.lat_home, self.lon_home, 0., lat_v, lon_v, 0.)
        return np.array([x_v, y_v, estate.z])

    def follow_the_carrot_reference(self, estates, path_parameter, T_carrot=25.):
        positions = [self._get_relative_position(e) for e in estates]
        vel_nsb, _ = self.step(positions, path_parameter)

        refs = []
        for i in range(len(vel_nsb)):
            v_i = vel_nsb[i]
            U = np.linalg.norm(v_i)
            p_i = positions[i]
            p_ref_i = p_i + T_carrot * v_i

            lat, lon = imcpy.coordinates.WGS84.displace(self.lat_home, self.lon_home, n=p_ref_i[0], e=p_ref_i[1])
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

    def simulated_response_reference(self, estates, path_parameter, T_sim=2.5, L_carrot=25.):    
        n = len(estates)
        pos0 = [self._get_relative_position(e) for e in estates]
        pos = [p.copy() for p in pos0]
        U = np.zeros(n)
        self._nsb_forward_euler(pos, path_parameter, U, T_sim, int(T_sim*10))
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

    def _nsb_forward_euler(self, positions, path_parameter, Ui, T, n_steps):
        param_copy = path_parameter
        for i in range(n_steps):
            param_copy = self._nsb_forward_euler_step(positions, param_copy, Ui,  T / n_steps)

    def _nsb_forward_euler_step(self, positions, path_parameter, Ui,dt):
        vel_nsb, param_derivative = self.step(positions, path_parameter)
        for i in range(len(positions)):
            positions[i] += vel_nsb[i] * dt
            Ui[i] += np.linalg.norm(vel_nsb[i]) * dt
        return path_parameter + param_derivative * dt
