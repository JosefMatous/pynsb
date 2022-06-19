from los import LineOfSight
from formation import FormationKeeping
from nsb import NSBAlgorithm
from geompath import PlanarSineWave, OscillatingEllipse

import configparser
import numpy as np

_DEFAULT_CONFIG = {
    'General' : {
        'Vehicles'                  : '[lauv-xplore-1, lauv-xplore-2, lauv-xplore-3]',
        'Home Latitude'             : '41.18500100646542',
        'Home Longitude'            : '-8.70620026716275',
        'Start Latitude'            : '[41.18520811, 41.18513607, 41.18513607]',
        'Start Longitude'           : '[-8.70618835, -8.7060334 , -8.70636713]',
        'Stop Latitude'             : '[41.18520811, 41.18513607, 41.18513607]',
        'Stop Longitude'            : '[-8.70618835, -8.7060334 , -8.70636713]',
        'Experiment Stop Time'      : '400.',
        'Experiment Stop Parameter' : '6.05',
        'Log Data'                  : 'True'
    },
    'Path' : {'Path Type' : 'Oscillating Ellipse'},
    'Planar Sine Wave' : {
        'Initial Point'  : '[25., 0., 0.]',
        'Amplitude'      : '15.0',
        'Wavenumber'     : '0.0419',
        'Path Yaw Angle' : '0.0'
    },
    'Oscillating Ellipse' : {
        'Center'              : '[25., -40., 1.5]',
        'Semimajor Axis'      : '70.0',
        'Semiminor Axis'      : '40.0',
        'Ellipse Orientation' : '0.0',
        'Initial Phase'       : '1.5707963267948966',
        'Clockwise'           : 'False',
        'Z Amplitude'         : '1.0',
        'Z Frequency'         : '2.0',
        'Z Initial Phase'     : '-1.5707963267948966'
    },
    'Line of Sight' : {
        'Lookahead Distance'          : '15.0',
        'Adaptive Lookahead Distance' : 'True',
        'Path Following Speed'        : '1.0',
        'Path Parameter Update Gain'  : '0.2'
    },
    'Formation Keeping' : {
        'Formation Shape'        : '[[6.0, 0.0, 0.0], [-3.0, 15.0, 0.0], [3.0, -15.0, 0.0]]',
        'Formation Keeping Gain' : '0.05'
    }
}

def load_configuration(config_file=None):
    cfg = configparser.ConfigParser()
    # Load the default configuration
    cfg.read_dict(_DEFAULT_CONFIG)
    # Load the given config file
    if config_file is not None:
        try:
            cfg.read(config_file)
        except:
            print('config_loader: could not read configuration from {}; will use default config instead'.format(config_file))
    
    # Create path object
    path = None
    path_type = cfg['Path']['Path Type']
    if path_type == 'Planar Sine Wave':
        try:
            path_dict = cfg['Planar Sine Wave']
            p0 = _str2ndarray(path_dict['Initial Position'])
            A = float(path_dict['Amplitude'])
            omega = float(path_dict['Wavenumber'])
            psi = float(path_dict['Path Yaw Angle'])
            path = PlanarSineWave(p0, A, omega, psi)
        except:
            raise ValueError('Error while reading configuration file')
    elif path_type == 'Oscillating Ellipse':
        try:
            path_dict = cfg['Oscillating Ellipse']
            c = _str2ndarray(path_dict['Center'])
            a = float(path_dict['Semimajor Axis'])
            b = float(path_dict['Semiminor Axis'])
            psi = float(path_dict['Ellipse Orientation'])
            phi0 = float(path_dict['Initial Phase'])
            clk = _str2bool(path_dict['Clockwise'])
            z_ampl = float(path_dict['Z Amplitude'])
            z_f = float(path_dict['Z Frequency'])
            z_phi0 = float(path_dict['Z Initial Phase'])
            path = OscillatingEllipse(c, a, b, psi, phi0, clk, z_ampl, z_f, z_phi0)
        except:
            raise ValueError('Error while reading configuration file')
    else:
        raise ValueError('Unknown path type "{}"'.format(path_type))

    # Create LOS object
    los_dict = cfg['Line of Sight']
    try:
        delta = float(los_dict['Lookahead Distance'])
        d_adapt = _str2bool(los_dict['Adaptive Lookahead Distance'])
        U = float(los_dict['Path Following Speed'])
        k = float(los_dict['Path Parameter Update Gain'])
        los = LineOfSight(delta, d_adapt, U, k)
    except:
        raise ValueError('Error while reading configuration file')

    # Create formation keeping object
    form_dict = cfg['Formation Keeping']
    try:
        shape = _str2arrarray(form_dict['Formation Shape'])
        k = float(form_dict['Formation Keeping Gain'])
        form = FormationKeeping([np.array(s) for s in shape], k)
    except:
        raise ValueError('Error while reading configuration file')

    # General configuration
    gen_dict = cfg['General']
    try:
        vehicles = [v.strip() for v in gen_dict['Vehicles'].strip('[ ]').split(',')]
        lat_home = np.deg2rad(float(gen_dict['Home Latitude']))
        lon_home = np.deg2rad(float(gen_dict['Home Longitude']))
        lat_start = np.deg2rad(_str2ndarray(gen_dict['Start Latitude']))
        lon_start = np.deg2rad(_str2ndarray(gen_dict['Start Longitude']))
        lat_stop = np.deg2rad(_str2ndarray(gen_dict['Stop Latitude']))
        lon_stop = np.deg2rad(_str2ndarray(gen_dict['Stop Longitude']))
        T_stop = float(gen_dict['Experiment Stop Time'])
        s_stop = float(gen_dict['Experiment Stop Parameter'])
        log_data = gen_dict['Log Data'].lower() == 'true'
        general = {'vehicles': vehicles,
                'lat_home': lat_home,
                'lon_home': lon_home,
                'lat_start': lat_start,
                'lon_start': lon_start,
                'lat_stop': lat_stop,
                'lon_stop': lon_stop,
                'T_stop': T_stop,
                's_stop': s_stop,
                'log_data': log_data}
    except:
        raise ValueError('Error while reading configuration file')

    return NSBAlgorithm(path, los, form, lat_home, lon_home), general

def _str2arrarray(s):
    """
    String to array of arrays
    """
    vals = [x.strip().strip('[').strip(']') for x in s.split('],')]
    return [_str2ndarray(x) for x in vals]

def _str2ndarray(s):
    """
    String to numpy array
    """
    return np.array([float(x) for x in _str2strarray(s)])

def _str2strarray(s):
    """
    String to array of strings
    """
    return [x.strip() for x in s.strip('[]').split(',')]

def _str2bool(s):
    """
    String to bool

    Returns true if the given string is "true" (case-insesitive)
    """
    return s.lower() == 'true'

if __name__ == "__main__":
    nsb, gen = load_configuration()
    print(gen)
