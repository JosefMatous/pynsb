import os, time

from nsb import NSBAlgorithm
import geompath

class DataLogger:
    def __init__(self, log_directory, alg: NSBAlgorithm):
        try:
            os.makedirs(os.path.dirname(log_directory), exist_ok=True)
            # Start path parameter log
            self.param_handle = open(os.path.join(log_directory, 'path_parameter.csv'), 'w')
            self.param_handle.write('timestamp, path_parameter\n')
            # Log NSB parameters
            self.config_handle = open(os.path.join(log_directory, 'configuration.ini'), 'w')
            ## Parameters of the path
            self._write_config_header('Path', False)
            if isinstance(alg.path, geompath.PlanarSineWave):
                self._write_config_line('Path Type', 'Planar Sine Wave')
                self._write_config_line('Initial Point', alg.path.p0)
                self._write_config_line('Amplitude', alg.path.amplitude)
                self._write_config_line('Wavenumber', alg.path.wavenumber)
                self._write_config_line('Path Yaw Angle', alg.path.yaw)
            elif isinstance(alg.path, geompath.OscillatingEllipse):
                self._write_config_line('Path Type', 'Oscillating Ellipse')
                self._write_config_line('Center', alg.path.center)
                self._write_config_line('Semimajor Axis', alg.path.a)
                self._write_config_line('Semiminor Axis', alg.path.b)
                self._write_config_line('Ellipse Orientation', alg.path.yaw)
                self._write_config_line('Initial Phase', alg.path.phi0)
                self._write_config_line('Clockwise', alg.path.clockwise)
                self._write_config_line('Z Amplitude', alg.path.z_amplitude)
                self._write_config_line('Z Frequency', alg.path.z_frequency)
                self._write_config_line('Z Initial Phase', alg.path.z_phi0)
            else:
                self._write_config_line('Path Type', 'Unknown Path Type')

            ## Line of sight guidance
            self._write_config_header('Line of Sight')
            self._write_config_line('Lookahead Distance', alg.los.lookahead)
            self._write_config_line('Adaptive Lookahead Distance', alg.los.adaptive)
            self._write_config_line('Path Following Speed', alg.los.speed)
            self._write_config_line('Path Parameter Update Gain', alg.los.gain)

            ## Formation keeping
            self._write_config_header('Formation Keeping')
            self._write_config_line('Formation Shape', [x.tolist() for x in alg.form.shape])
            self._write_config_line('Formation Keeping Gain', alg.form.gain)
            self.config_handle.close()
            self.config_handle = None
        except:
            print('Could not create log files')
            self.config_handle = None
            self.param_handle = None

    def _write_config_header(self, header, skipline=True):
        if skipline:
            self.config_handle.write('\n')
        self.config_handle.write('[{}]\n'.format(header))

    def _write_config_line(self, key, value):
        self.config_handle.write('{:<30} = {}\n'.format(key, value))

    def log_path_parameter(self, param):
        if self.param_handle is not None:
            self.param_handle.write('{}, {}\n'.format(time.time(), param))

    def close(self):
        if self.param_handle is not None:
            self.param_handle.close()
            self.param_handle = None
        if self.config_handle is not None:
            self.config_handle.close()
            self.config_handle = None
