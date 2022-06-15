import geompath, nsb, los
from los import LineOfSight
from formation import FormationKeeping
from utilities import calculate_barycenter, predict_vehicle_state

import logging, sys
import numpy as np

import imcpy
from imcpy.actors import DynamicActor
from imcpy.decorators import Subscribe, Periodic

class NSBAlgorithm(DynamicActor):
    def __init__(self, vehicles, path, los, form):
        super().__init__()
        self.vehicles = vehicles
        self.estates = {v : None for v in vehicles}
        for v in vehicles:
            self.heartbeat.append(v)
        self.path = path
        self.path_parameter = 0.
        self.los = los
        self.form = form
        self.started = False

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

    def has_valid_estimates(self):
        valid = [e is not None for e in self.estates.values()]
        return all(valid)

    @Periodic(0.1)
    def update_path_parameter(self):
        if self.has_valid_estimates() and self.started:
            path_ref = self.path.getPathReference(self.path_parameter)
            positions = [np.array([e.x, e.y, e.z]) for e in self.estates.values()]
            _, param_derivative = self.los.step(path_ref, calculate_barycenter(positions))
            self.path_parameter += param_derivative * 0.1

    @Periodic(5)
    def send_references(self):
        if self.has_valid_estimates():
            # Update state estimates
            for e in self.estates.values():
                predict_vehicle_state(e)
            refs = nsb.simulated_response_reference(list(self.estates.values()), self.path_parameter, self.path, self.los, self.form)
            for i in range(len(refs)):
                id = self.resolve_node_id(self.vehicles[i])
                self.send(id, refs[i])
            self.started = True


if __name__ == '__main__':
    # Setup logging level and console output
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Run actor
    x = NSBAlgorithm(sys.argv[1:], geompath.PlanarSineWave(), LineOfSight(adaptive=True), FormationKeeping())
    x.run()
