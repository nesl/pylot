import time
import params
import socket
import pickle
import csv

from utils.logging import setup_pipeline_logging, ModuleCompletionLogger
from utils.simulation import get_world
from objects.messages import SensorMessage

from simulation import CarlaSimulation
from visualizer import Visualizer
from detection.object_detection import ObjectDetector
from perception.object_tracking import ObjectTracker
from perception.location_history import ObstacleLocationHistory
from objects.messages import ObstacleTrajectoriesMessage
from planning.planner import WaypointPlanner
from control.controller import Controller

from prediction.predictor import get_predictions
from utils.service import send_msg, recv_msg

class SimulationRunner():

    def __init__(self):

        client, world = get_world(params.simulator_host,
                             params.simulator_port,
                             params.simulator_timeout)

        self._simulation = CarlaSimulation(client, world)
        self._detector = ObjectDetector()
        self._tracker = ObjectTracker()
        self._history = ObstacleLocationHistory()
        # self._predictor = ...
        self._planner = WaypointPlanner(world.get_map())
        self._controller = Controller()
        self._visualizer = Visualizer(world)

        self.throttle = -1
        self.brake = -1
        self.steer = -1

        self.csvfile = open('planner_dump.csv', 'w', newline='')
        self.csvwriter = csv.writer(self.csvfile)
        self.csvwriter.writerow(['timestamp', 'pose', 'waypoints'])

    def run_one_tick(self):
        (timestamp, frame, depth_frame, pose) = self._simulation.tick_simulator()
        if not frame:
            print("Empty frame received from simulation!")
            return
        
        obstacles = []
        tracked_obstacles = []
        obstacle_trajectories = []
        obstacle_predictions = []
        waypoints = None

        (timestamp, obstacles, detector_runtime) = self._detector.get_obstacles(timestamp, frame)
        print("Detected obstacles {} {}".format(len(obstacles), detector_runtime))
        
        (timestamp, tracked_obstacles, tracker_runtime) = self._tracker.get_tracked_obstacles(timestamp, frame, obstacles)
        print("Tracked obstacles  {} {}".format(len(obstacles), tracker_runtime))
        
        if len(tracked_obstacles) > 0:
            (timestamp, obstacle_trajectories) = self._history.get_location_history(timestamp, pose, depth_frame, tracked_obstacles)
            print("Trajectories       {} ".format(len(obstacle_trajectories)))
        
        if len(obstacle_trajectories) > 0:
            obstacle_trajectories_message = ObstacleTrajectoriesMessage(obstacle_trajectories) # necessary because this contains methods used in prediction
            (obstacle_predictions, predictor_runtime) = get_predictions(obstacle_trajectories_message)
            print("Predictions        {} {}".format(len(obstacle_predictions), predictor_runtime))
        
        if len(obstacle_predictions) > 0:
            print(pose)
            (waypoints, planner_runtime) = self._planner.get_waypoints(timestamp, pose, obstacle_predictions)
            print("Planner waypoints  {} {}".format(len(waypoints.waypoints), planner_runtime))
            self.csvwriter.writerow([timestamp, pose, waypoints.waypoints])
        
        (steer, throttle, brake, controller_runtime) = self._controller.get_control_instructions(timestamp, pose, waypoints)
        print("Control instructions {} {} {} {}".format(throttle, steer, brake, controller_runtime))

        if throttle == 0 and brake == 0.5 and steer == 0:
            if self.throttle != -1 and self.brake != -1:
                throttle = self.throttle
                brake = self.brake
                steer = self.steer

        self.throttle = throttle
        self.brake = brake
        self.steer = steer            

        self._simulation.apply_control(throttle, steer, brake, False, False)
        self._visualizer.visualize(timestamp, frame, depth_frame, pose, obstacles, 1, 0, 0)

class MockSimulationRunner():
    
    def __init__(self):
        client, world = get_world(params.simulator_host,
                             params.simulator_port,
                             params.simulator_timeout)
        self._simulation = CarlaSimulation(client, world)
        self._visualizer = Visualizer(world)
        self._detector = ObjectDetector()
        self._tracker = ObjectTracker()
        self._history = ObstacleLocationHistory()
        self._controller = Controller()
        self._planner = WaypointPlanner(world.get_map())
        
        if params.distributed == True:
            self.sim_connect_to_server()

    def sim_connect_to_server(self):
        # Simulator will always send data to local computer i.e. RasPi
        host = params.local_server
        port = params.local_port
        
        self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._s.connect((host, port))
    
    def run_one_tick(self):
        (timestamp, frame, depth_frame, pose) = self._simulation.tick_simulator()
        if not frame:
            print("Empty frame received from simulation!")
            return

        if params.distributed == True:
            sensor_data = SensorMessage(timestamp=timestamp, frame=frame, depth_frame=depth_frame, pose=pose)
            sdd = pickle.dumps(sensor_data)
            send_msg(self._s, sdd)
            print("----------------------Sent sensor data"+str(len(sdd)))
            control_msg = recv_msg(self._s)
            print("Rxvd control message"+str(len(control_msg)))
            control_msg = pickle.loads(control_msg)
            self._simulation.apply_control(control_msg.throttle, control_msg.steer, control_msg.brake, False, False)
            self._visualizer.visualize(timestamp, frame, depth_frame, pose, None, control_msg.throttle, control_msg.steer, control_msg.brake)


        else:
            obstacles = []
            tracked_obstacles = []
            obstacle_trajectories = []
            obstacle_predictions = []
            waypoints = None

            (timestamp, obstacles, detector_runtime) = self._detector.get_obstacles(timestamp, frame)
            print("Detected obstacles {} {}".format(len(obstacles), detector_runtime))
            
            (timestamp, tracked_obstacles, tracker_runtime) = self._tracker.get_tracked_obstacles(timestamp, frame, obstacles)
            print("Tracked obstacles  {} {}".format(len(obstacles), tracker_runtime))
            
            if len(tracked_obstacles) > 0:
                (timestamp, obstacle_trajectories) = self._history.get_location_history(timestamp, pose, depth_frame, tracked_obstacles)
                print("Trajectories       {} ".format(len(obstacle_trajectories)))
            
            if len(obstacle_trajectories) > 0:
                obstacle_trajectories_message = ObstacleTrajectoriesMessage(obstacle_trajectories) # necessary because this contains methods used in prediction
                (obstacle_predictions, predictor_runtime) = get_predictions(obstacle_trajectories_message)
                print("Predictions        {} {}".format(len(obstacle_predictions), predictor_runtime))
            
            if len(obstacle_predictions) > 0:
                print("Predictions  - " + str(obstacle_predictions[0]))
                (waypoints, planner_runtime) = self._planner.get_waypoints(pose, obstacle_predictions)
                print("Planner waypoints  {} {}".format(len(waypoints.waypoints), planner_runtime))
            
            (steer, throttle, brake, controller_runtime) = self._controller.get_control_instructions(timestamp, pose, waypoints)
            print("Control instructions {} {} {} {}".format(throttle, steer, brake, controller_runtime)) 
            
            self._simulation.apply_control(throttle, steer, brake, False, False)
            self._visualizer.visualize(timestamp, frame, depth_frame, pose, obstacles, throttle, steer, brake)

def main():
    setup_pipeline_logging()
    runner = SimulationRunner()
    while True:
        runner.run_one_tick()

if __name__ == '__main__':
    main()
