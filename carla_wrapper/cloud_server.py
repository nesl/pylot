import socket
import pickle
import threading
import time
import params
import zlib
import numpy as np

from collections import deque

from detection.object_detection import ObjectDetector
from perception.object_tracking import ObjectTracker
from perception.location_history import ObstacleLocationHistory
from objects.messages import ObstacleTrajectoriesMessage
from planning.planner import WaypointPlanner
from control.controller import Controller

from objects.messages import SensorMessage, ControlMessage, PlannerMessage
from prediction.predictor import get_predictions
from utils.service import send_msg, recv_msg

def cloud_server():
    host = params.cloud_server
    port = params.cloud_port

    cloud_socket = socket.socket() 
    cloud_socket.bind((host, port))

    detector = ObjectDetector()
    tracker = ObjectTracker()
    history = ObstacleLocationHistory()
    controller = Controller()
    planner = WaypointPlanner(None)

    print("Cloud server open at..", host, port)

    # configure how many client the server can listen simultaneously
    cloud_socket.listen(10)
    cloud_conn, address = cloud_socket.accept()  

    while True:

        if params.perception_loc == 'cloud':
            start_time = time.time()
            input = recv_msg(cloud_conn)
            end_time = (time.time()-start_time)
            print("rx sensor data: " + str(end_time))
            time1 = time.time()
            input_message = pickle.loads(input)
            sensor_data = SensorMessage(timestamp = input_message.timestamp,
                                        frame=input_message.frame,
                                        depth_frame=input_message.depth_frame,
                                        pose=input_message.pose,
                                        local_send_time = input_message.local_send_time)

            obstacles = []
            tracked_obstacles = []
            obstacle_trajectories = []
            obstacle_predictions = []
            waypoints = None

            timestamp = input_message.timestamp

            sensor_data.frame.frame = zlib.decompress(sensor_data.frame.frame)
            sensor_data.frame.frame = np.frombuffer(sensor_data.frame.frame, dtype=np.uint8)
            sensor_data.frame.frame = np.reshape(sensor_data.frame.frame, (512, 960, 3))
            sensor_data.depth_frame.frame = zlib.decompress(sensor_data.depth_frame.frame)
            sensor_data.depth_frame.frame = np.frombuffer(sensor_data.depth_frame.frame, dtype='float32')
            #depth_frame.frame = depth_frame.frame.astype(np.float16)
            sensor_data.depth_frame.frame = np.reshape(sensor_data.depth_frame.frame, (512, 960))
            print("Calculated message proc runtime = ", time.time()-time1)
            time2 = time.time()
            (timestamp, obstacles, detector_runtime) = detector.get_obstacles(timestamp, sensor_data.frame)
            print("Detected obstacles {} {}".format(len(obstacles), detector_runtime))
            print("Calculated detector runtime = ", time.time()-time2)
            
            time3 = time.time()
            (timestamp, tracked_obstacles, tracker_runtime) = tracker.get_tracked_obstacles(timestamp, sensor_data.frame, obstacles)
            #print("Tracked obstacles  {} {}".format(len(obstacles), tracker_runtime))
            print("Calculated tracker runtime = ", time.time()-time3)
        
            
            if len(tracked_obstacles) > 0:
                time4 = time.time()
                (timestamp, obstacle_trajectories) = history.get_location_history(timestamp, sensor_data.pose, sensor_data.depth_frame, tracked_obstacles)
                # print("Trajectories       {} ".format(len(obstacle_trajectories)))
                print("Calculated location history runtime = ", time.time()-time4)
            
            if len(obstacle_trajectories) > 0:
                first_trajectory = obstacle_trajectories[0].trajectory
                # for traj_location in first_trajectory:
                #     print("Trajectory 1  - " + str(traj_location))
                time5 = time.time()
                obstacle_trajectories_message = ObstacleTrajectoriesMessage(obstacle_trajectories) # necessary because this contains methods used in prediction
                (obstacle_predictions, predictor_runtime) = get_predictions(obstacle_trajectories_message)
                #print("Predictions        {} {}".format(len(obstacle_predictions), predictor_runtime))
                print("Calculated prediction runtime = ", time.time()-time5)
            
            if len(obstacle_predictions) > 0:
                # print("Predictions  - " + str(obstacle_predictions[0]))
                time6 = time.time()
                (waypoints, planner_runtime) = planner.get_waypoints(timestamp, sensor_data.pose, obstacle_predictions)
                #print("Planner waypoints  {} {}".format(len(waypoints.waypoints), planner_runtime))
                print("Calculated planner runtime = ", time.time()-time6)

            if params.control_loc == 'cloud':
                time7 = time.time()
                (steer, throttle, brake, controller_runtime) = controller.get_control_instructions(timestamp, sensor_data.pose, waypoints)
                print("Calculated control runtime = ", time.time()-time7)
                print("Control instructions {} {} {} {}".format(throttle, steer, brake, controller_runtime))
                control_msg = ControlMessage(steer=steer, throttle=throttle, brake=brake, hand_brake=False, reverse=False, timestamp=timestamp)
                print("Calculated end to end runtime = ", time.time()-time1)
                send_msg(cloud_conn, pickle.dumps(control_msg))
                print("Sent control message")

            elif params.control_loc == 'local':
                planner_msg = PlannerMessage(timestamp=timestamp, pose=sensor_data.pose, waypoints=waypoints)
                pmsg = pickle.dumps(planner_msg)
                print("Len of planner message: ", len(pmsg))
                send_msg(cloud_conn, pmsg)
                print("Sent planner message")

        elif params.perception_loc == 'local' and params.control_loc == 'cloud':
            input = recv_msg(cloud_conn)
            input_message = pickle.loads(input)
            if input_message != None and input_message.waypoints != None and len(input_message.waypoints.waypoints) > 10:
                input_message.waypoints.waypoints = deque([input_message.waypoints.waypoints[i] for i in range(0, 10, 1)])
            (steer, throttle, brake, controller_runtime) = controller.get_control_instructions(input_message.timestamp, input_message.pose, input_message.waypoints)
            print("Control instructions {} {} {} {}".format(throttle, steer, brake, controller_runtime))
            control_msg = ControlMessage(steer=steer, throttle=throttle, brake=brake, hand_brake=False, reverse=False, timestamp=0)
            send_msg(cloud_conn, pickle.dumps(control_msg))
            print("Sent control message")

    #     print("Generated tracker message: ", tracker_message)
    #     conn.send(pickle.dumps(tracker_message))  # send data to the client
    # conn.close()  # close the connection


if __name__=='__main__':
    #thread_one = threading.Thread(target=controller_server)
    #thread_two = threading.Thread(target=tracker_server)
    #thread_one.start()
    #thread_two.start()
    cloud_server()
