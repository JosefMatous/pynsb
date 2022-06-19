from enum import Enum
import geompath
from los import LineOfSight
from formation import FormationKeeping
from utilities import calculate_barycenter, predict_vehicle_state
from nsb import NSBAlgorithm
from data_logger import DataLogger
from config_loader import load_configuration

import logging, sys, time, os
import numpy as np

import imcpy
from imcpy.actors import DynamicActor
from imcpy.decorators import Subscribe, Periodic

logger = logging.getLogger('nsb-task')

class TaskState(Enum):
    INITIAL = 0
    PLAN_SENT = 2
    GOTO_START = 3
    STARTING_EXPERIMENT = 4
    RUN_EXPERIMENT = 5
    GOTO_STOP = 6
    STOPPED = 7

class NSBTask(DynamicActor):
    def __init__(self, vehicles, nsb: NSBAlgorithm,
                lat_start=None, lon_start=None, lat_stop=None, lon_stop=None, 
                T_stop=200., s_stop=None,
                log_data=False, log_directory=None):
        super().__init__()
        self.vehicles = vehicles
        self.estates = {v : None for v in vehicles}
        self.near_target = {v : False for v in vehicles}
        for v in vehicles:
            self.heartbeat.append(v)
        self.path = nsb.path
        self.path_parameter = 0.
        self.los = nsb.los
        self.state = TaskState.INITIAL
        self.nsb = nsb
        if lat_start is None:
            self.lat_start = [0.7188174845559216, 0.7188162273169989, 0.7188162273169989]
        else:
            self.lat_start = lat_start
        if lon_start is None:
            self.lon_start = [-0.15195165197315574, -0.15194894762736014, -0.15195477237263982]
        else:
            self.lon_start = lon_start
        if lat_stop is None:
            self.lat_stop = [0.7188174845559216, 0.7188162273169989, 0.7188162273169989]
        else:
            self.lat_stop = lat_stop
        if lon_stop is None:
            self.lon_stop = [-0.15195165197315574, -0.15194894762736014, -0.15195477237263982]
        else:
            self.lon_stop = lon_stop
        self.timestamp_start = None
        self.T_stop = T_stop if T_stop is not None else np.inf
        self.s_stop = s_stop if s_stop is not None else np.inf
        if T_stop is None and s_stop is None:
            raise ValueError('Both experiment stop time and stop parameter cannot be None.')
        self.log_data = log_data
        if log_data:
            if log_directory is None:
                log_directory = os.getcwd()
            self.data_logger = DataLogger(log_directory, self.nsb)       

    def get_source(self, msg):
        try:
            node = self.resolve_node_id(msg)
            return node.sys_name
        except KeyError:
            return None

    @Subscribe(imcpy.EstimatedState)
    def recv_estate(self, msg):
        vehicle = self.get_source(msg)
        if vehicle in self.vehicles:
            self.estates[vehicle] = msg.clone()

    @Subscribe(imcpy.FollowRefState)
    def recv_followrefstate(self, msg: imcpy.FollowRefState):
        src = self.get_source(msg)
        if src not in self.vehicles:
            return

        if self.state in (TaskState.GOTO_START, TaskState.RUN_EXPERIMENT, TaskState.GOTO_STOP):
            if msg.state == imcpy.FollowRefState.StateEnum.GOTO:
                # In goto maneuver
                if msg.proximity & imcpy.FollowRefState.ProximityBits.XY_NEAR:
                    # Near XY
                    self.near_target[src] = True
                    self._handle_target_update()
            elif msg.state in (imcpy.FollowRefState.StateEnum.LOITER, imcpy.FollowRefState.StateEnum.HOVER, imcpy.FollowRefState.StateEnum.WAIT):
                # Loitering/hovering/waiting
                self.near_target[src] = True
                self._handle_target_update()

    def _handle_target_update(self):
        if all(self.near_target.values()):
            self._clear_near_target()
            if self.state == TaskState.GOTO_START:
                self.state = TaskState.RUN_EXPERIMENT
                logger.info('Starting experiment')
            elif self.state == TaskState.GOTO_STOP:
                self.stop_plan()
            elif self.state == TaskState.RUN_EXPERIMENT:
                logger.warn('Vehicles reached destination during experiment')
                self.send_references()

    def _clear_near_target(self):
        for v in self.vehicles:
            self.near_target[v] = False

    def has_valid_estimates(self):
        valid = [e is not None for e in self.estates.values()]
        return all(valid)

    @Periodic(0.1)
    def update_path_parameter(self):
        if self.state == TaskState.RUN_EXPERIMENT and self.timestamp_start is not None:
            path_ref = self.path.getPathReference(self.path_parameter)
            positions = [np.array([e.x, e.y, e.z]) for e in self.estates.values()]
            _, param_derivative = self.los.step(path_ref, calculate_barycenter(positions))
            self.path_parameter += param_derivative * 0.1
            if self.log_data:
                self.data_logger.log_path_parameter(self.path_parameter)

    @Periodic(5)
    def main_task(self):
        if self.has_valid_estimates():
            if self.state == TaskState.INITIAL:
                self.send_plan()
            elif self.state == TaskState.PLAN_SENT:
                self.state = TaskState.GOTO_START
            elif self.state == TaskState.GOTO_START:
                self.send_goto_start()
            elif self.state == TaskState.RUN_EXPERIMENT:
                self.send_references()
            elif self.state == TaskState.GOTO_STOP:
                self.send_goto_stop()
                if self.log_data:
                    self.data_logger.close()

    def send_plan(self):
        for vehicle in self.vehicles:
            node = self.resolve_node_id(vehicle)
            fr = imcpy.FollowReference()
            fr.control_src = 0xFFFF  # Controllable from all IMC adresses
            fr.control_ent = 0xFF  # Controllable from all entities
            fr.timeout = 30.0  # Maneuver stops when time since last Reference message exceeds this value
            fr.loiter_radius = 5.  # Default loiter radius when waypoint is reached
            fr.altitude_interval = 0.

            # Add to PlanManeuver message
            pman = imcpy.PlanManeuver()
            pman.data = fr
            pman.maneuver_id = '1'

            # Add to PlanSpecification
            spec = imcpy.PlanSpecification()
            spec.plan_id = 'follow_nsb'
            spec.maneuvers.append(pman)
            spec.start_man_id = '1'
            spec.description = 'follow references from NSB algorithm'

            # Start plan
            pc = imcpy.PlanControl()
            pc.type = imcpy.PlanControl.TypeEnum.REQUEST
            pc.op = imcpy.PlanControl.OperationEnum.START
            pc.plan_id = 'follow_nsb'
            pc.arg = spec

            self.send(node, pc)

            self.state = TaskState.PLAN_SENT
        logger.info('Started FollowRef command')

    def stop_plan(self):
        self._send_goto(self.lat_stop, self.lon_stop, True)
        logger.info('Stopped FollowRef command')
        self.state = TaskState.STOPPED
        self.stop()

    def _send_goto(self, lat, lon, final=False):
        for i in range(len(lat)):
            node = self.resolve_node_id(self.vehicles[i])
            r = imcpy.Reference()
            r.lat = lat[i]  # Target waypoint
            r.lon = lon[i]  # Target waypoint
            
            # Assign z
            dz = imcpy.DesiredZ()
            dz.value = 0.0
            dz.z_units = imcpy.ZUnits.DEPTH
            r.z = dz

            # Assign the speed
            ds = imcpy.DesiredSpeed()
            ds.value = 1.
            ds.speed_units = imcpy.SpeedUnits.METERS_PS
            r.speed = ds

            # Bitwise flags (see IMC spec for explanation)
            flags = imcpy.Reference.FlagsBits.LOCATION | imcpy.Reference.FlagsBits.SPEED | imcpy.Reference.FlagsBits.Z
            flags = flags | imcpy.Reference.FlagsBits.MANDONE if final else flags
            r.flags = flags
            #logger.info('Sending goto to {}'.format(self.vehicles[i]))
            self.last_ref = r
            self.send(node, r)

    def send_goto_start(self):
        self._send_goto(self.lat_start, self.lon_start)
        logger.info('Going to start position')
    
    def send_goto_stop(self):
        self._send_goto(self.lat_stop, self.lon_stop)
        logger.info('Going to stop position')

    def send_references(self):
        if self.timestamp_start is None:
            self.timestamp_start = time.time()
        t_diff = time.time() - self.timestamp_start
        logger.info('Experiment in progress (t = {}s)'.format(t_diff))
        if t_diff >= self.T_stop or self.path_parameter >= self.s_stop:
            self.state = TaskState.GOTO_STOP
            logger.info('Stopping experiment')
        # Update state estimates
        for e in self.estates.values():
            predict_vehicle_state(e)
        refs = self.nsb.simulated_response_reference(list(self.estates.values()), self.path_parameter)
        for i in range(len(refs)):
            id = self.resolve_node_id(self.vehicles[i])
            self.send(id, refs[i])

    def stop(self):
        super().stop()
        print('Stopping task')
        if self.log_data:
            self.data_logger.close()

if __name__ == '__main__':
    # Setup logging level and console output
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Read configuration file
    config_file = None
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    nsb, gen = load_configuration(config_file)

    # Run actor
    scirpt_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = 'log/' + time.strftime('%Y%m%d/%H%M%S/', time.gmtime())
    x = NSBTask(gen['vehicles'], nsb, lat_start=gen['lat_start'], lon_start=gen['lon_start'],
            lat_stop=gen['lat_stop'], lon_stop=gen['lon_stop'], 
            T_stop=gen['T_stop'], s_stop=gen['s_stop'],
            log_data=gen['log_data'], log_directory=os.path.join(scirpt_dir, log_dir))
    x.run()
