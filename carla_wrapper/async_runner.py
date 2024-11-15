import time
import threading
import params
import copy
import socket
import pickle
import zlib
import numpy as np
from utils.logging import setup_pipeline_logging, ModuleCompletionLogger
from utils.simulation import get_world

from simulation import CarlaSimulation
from visualizer import Visualizer
#from detection.object_detection import ObjectDetector
from perception.object_tracking import ObjectTracker
from perception.location_history import ObstacleLocationHistory
from objects.messages import ObstacleTrajectoriesMessage
from planning.planner import WaypointPlanner
from control.controller import Controller

from prediction.predictor import get_predictions

from utils.service import send_msg, recv_msg

from objects.messages import SensorMessage, ControlMessage

class AsyncSimulationRunner():
    
    def __init__(self):
        client, world = get_world(params.simulator_host,
                             params.simulator_port,
                             params.simulator_timeout)
        self._simulation = CarlaSimulation(client, world)
        self._visualizer = Visualizer(world)
        #self._detector = ObjectDetector()
        self._tracker = ObjectTracker()
        self._history = ObstacleLocationHistory()
        self._controller = Controller()
        self._planner = WaypointPlanner(world.get_map())

        # Shared data objects
        self._sim_lock = threading.Lock()
        self._control_lock = threading.Lock()
        self._pose = None
        self._frame = None
        self._depth_frame = None
        self._timestamp = -1

        self._throttle = -1
        self._brake = -1
        self._steer = -1
        self._control_timestamp = -1

        if params.distributed == True:
            self._server = self.connect_to_server()

    def connect_to_server(self):
        # Simulator will always send data to local computer i.e. RasPi
        host = params.local_server
        port = params.local_port
        
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.connect((host, port))
        return server
    
    def run(self):
        simulation = threading.Thread(target=self.run_simulation)
        pipeline = threading.Thread(target=self.run_pipeline)
        simulation.start()
        pipeline.start()

    # Tick at a specific cadence and constantly update world representation in parallel
    def run_simulation(self):
        throttle = 0
        brake = 0
        steer = 0
        control_timestamp = 0
        for _ in range(0, 15, 1): # needed to kick off the scenario
            self._simulation.tick_simulator()
            self._simulation.apply_control(0.5, 0, 0, False, False)
        #self._simulation.apply_control(1, 0, 1, False, False)
        while True:
            time.sleep(1.0 / params.simulator_fps) # Sleep for one frame
            timestamp, frame, depth_frame, pose = self._simulation.tick_simulator()
            with self._sim_lock:
                self._pose = pose
                self._frame = frame
                self._depth_frame = depth_frame
                self._timestamp = timestamp
            
            # Not visualizing obstacles or control params
            self._visualizer.visualize(timestamp, frame, depth_frame, pose, None, throttle, steer, brake)

            # if timestamp % 1000 == 0:
            #     self._simulation.apply_control(1, 0, 0, False, False)

            with self._control_lock:
                if control_timestamp == self._control_timestamp: # already processed
                    #print("Already processed control timestamp " + str(control_timestamp))
                    continue
                if self._throttle == -1 or self._brake == -1 or self._control_timestamp == -1:
                    #print("throttle or brake is -1 " + str(timestamp))
                    continue
                throttle = self._throttle
                brake = self._brake
                steer = self._steer
                control_timestamp = self._control_timestamp
            
            self._simulation.apply_control(throttle, steer, brake, False, False)
    
    def run_pipeline(self):
        pose = None
        frame = None
        depth_frame = None
        timestamp = 0
        while True:
            with self._sim_lock:
                if timestamp == self._timestamp: # already processed
                    #print("Already processed sim timestamp " + str(timestamp))
                    continue
                if self._pose == None or self._frame == None or self._depth_frame == None or self._timestamp == -1:
                    print("Empty pose or frame or depth frame " + str(timestamp))
                    continue
                pose = self._pose
                frame = self._frame
                depth_frame = self._depth_frame
                timestamp = self._timestamp
                sensor_data = SensorMessage(timestamp=timestamp, frame=frame, depth_frame=depth_frame, pose=pose)
                #print("Size of sensor data: ", len(sensor_data))
            fshape = frame.frame.shape
            dfshape = depth_frame.frame.shape
            dftype = depth_frame.frame.dtype

            if self._pose is not None or self._frame is not None or self._depth_frame is not None:
                frame.frame = zlib.compress(frame.frame)
                depth_frame.frame = zlib.compress(depth_frame.frame)
                pickle_pose = pickle.dumps(pose)
                pickle_frame = pickle.dumps(frame)
                pickle_depth_frame = pickle.dumps(depth_frame)
                pickle_timestamp = pickle.dumps(timestamp)
                sensor_data = SensorMessage(timestamp=timestamp, frame=frame, depth_frame=depth_frame, pose=pose)
                sdd = pickle.dumps(sensor_data)
                print("----------------------Sensor data: "+str(len(sdd)))
                print("----------------------Sensor pose: "+str(len(pickle_pose)))
                print("----------------------Sensor frame: "+str(len(pickle_frame)))
                print("----------------------Sensor depth frame: "+str(len(pickle_depth_frame)))
                print("----------------------Sensor timestamp: "+str(len(pickle_timestamp)))

            if params.distributed == True:
                sensor_data = SensorMessage(timestamp=timestamp, frame=frame, depth_frame=depth_frame, pose=pose)
                send_time = time.time()
                sensor_data.local_send_time = send_time
                sdd = pickle.dumps(sensor_data)
                send_msg(self._server, sdd)
                print("Time taken to send "+str(time.time()-send_time))
                print("----------------------Sent sensor data "+str(len(sdd)))
                control_msg = recv_msg(self._server)
                print("Rxvd control message "+str(len(control_msg)))
                rx_time = time.time()
                print("Time taken for sense-effect "+str(time.time()-send_time))
                control_msg = pickle.loads(control_msg)
                #print("Control message rx time: ", rx_time-control_msg.local_send_time)   
                with self._control_lock:
                    self._throttle = control_msg.throttle
                    self._brake = control_msg.brake
                    self._steer = control_msg.steer
                    self._control_timestamp = timestamp
                continue
            
            obstacle_trajectories = []
            obstacle_predictions = []
            waypoints = None

            frame.frame = zlib.decompress(frame.frame)
            print(fshape)
            frame.frame = np.frombuffer(frame.frame, dtype=np.uint8)
            frame.frame = np.reshape(frame.frame, (512,960,3))
            depth_frame.frame = zlib.decompress(depth_frame.frame)
            print(dfshape)
            print(dftype)
            depth_frame.frame = np.frombuffer(depth_frame.frame, dtype=dftype)
            #depth_frame.frame = depth_frame.frame.astype(np.float32)
            depth_frame.frame = np.reshape(depth_frame.frame, (512,960))
            (timestamp, obstacles, detector_runtime) = self._detector.get_obstacles(timestamp, frame)
            print("Detected obstacles {} {} {}".format(len(obstacles), detector_runtime, obstacles))
            
            (timestamp, tracked_obstacles, tracker_runtime) = self._tracker.get_tracked_obstacles(timestamp, frame, obstacles)
            print("Tracked obstacles  {} {} {}".format(len(obstacles), tracker_runtime, tracked_obstacles))
            
            if len(tracked_obstacles) > 0:
                (timestamp, obstacle_trajectories) = self._history.get_location_history(timestamp, pose, depth_frame, tracked_obstacles)
                print("Trajectories       {} ".format(len(obstacle_trajectories)))
            
            if len(obstacle_trajectories) > 0:
                #first_trajectory = obstacle_trajectories[0].trajectory
                # for traj_location in first_trajectory:
                #     print("Trajectory 1  - " + str(traj_location))
                obstacle_trajectories_message = ObstacleTrajectoriesMessage(obstacle_trajectories) # necessary because this contains methods used in prediction
                (obstacle_predictions, predictor_runtime) = get_predictions(obstacle_trajectories_message)
                print("Predictions        {} {}".format(len(obstacle_predictions), predictor_runtime))
            
            if len(obstacle_predictions) > 0:
                print("Predictions  - " + str(obstacle_predictions[0]))
                (waypoints, planner_runtime) = self._planner.get_waypoints(timestamp, pose, obstacle_predictions)
                print("Planner waypoints  {} {} {}".format(len(waypoints.waypoints), planner_runtime, waypoints))
            
            (steer, throttle, brake, controller_runtime) = self._controller.get_control_instructions(timestamp, pose, waypoints)
            print("Control instructions {} {} {} {}".format(throttle, steer, brake, controller_runtime))
            
            with self._control_lock:
                self._throttle = throttle
                self._brake = brake
                self._steer = steer
                self._control_timestamp = timestamp
    
def main():
    runner = AsyncSimulationRunner()
    runner.run()

if __name__ == '__main__':
    main()
