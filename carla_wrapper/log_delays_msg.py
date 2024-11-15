import time
import socket
import pickle
import struct
import numpy as np

from objects.messages import SensorMessage, ControlMessage, PlannerMessage
from prediction.predictor import get_predictions
from utils.service import send_msg, recv_msg
from objects.frames import CameraFrame, DepthFrame
from objects.objects import CameraSetup, DepthCameraSetup, Transform, Location, Rotation, Pose, Vector3D, Waypoints

def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def connect_to_server():
    host = '50.18.72.199'
    port = 10000

    cloud_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cloud_conn.connect((host, port))
    print("Connected to cloud..")

    while True:

        sensor_data = SensorMessage(timestamp = 8357,
                                        frame=CameraFrame(encoding= 'BGR', 
                                                            camera_setup= CameraSetup(name= 'center_camera', 
                                                            type= 'sensor.camera.rgb', 
                                                            width= 1024, 
                                                            height= 576, 
                                                            transform= Transform(location= Location(x=1.3, y=0.0, z=1.8), rotation= Rotation(pitch=-15, yaw=0, roll=0)), 
                                                            fov= 90.0)),
                                        depth_frame=DepthFrame(camera_setup= CameraSetup(name= 'depth_center_camera', 
                                                            type= 'sensor.camera.depth', 
                                                            width= 1024, 
                                                            height= 576, 
                                                            transform= Transform(location= Location(x=1.3, y=0.0, z=1.8), rotation= Rotation(pitch=-15, yaw=0, roll=0)), 
                                                            fov= 90.0)),
                                        pose=Pose(transform= Transform(location= Location(x=396.0, y=275.0, z=0.032674599438905716), rotation= Rotation(pitch=0.0, yaw=-89.99999237060547, roll=0.0)),
                                                            forward_speed= 0.0, 
                                                            velocity_vector= Vector3D(x=0.0, y=0.0, z=0.0)))
        send_msg(cloud_conn, pickle.dumps(sensor_data))
        print("Sensor data sent")

        # planner_data = PlannerMessage(pose=Pose(transform= Transform(location= Location(x=396.0, y=275.0, z=0.032674599438905716), rotation= Rotation(pitch=0.0, yaw=-89.99999237060547, roll=0.0)),
        #                                                     forward_speed= 0.0, 
        #                                                     velocity_vector= Vector3D(x=0.0, y=0.0, z=0.0)),
        #                                 waypoints=Waypoints(waypoints=[Transform(location= Location(x=396.0, y=275.0, z=0.032674599438905716), rotation= Rotation(pitch=0.0, yaw=-89.99999237060547, roll=0.0)),
        #                                                                 Transform(location= Location(x=396.0, y=275.0, z=0.032674599438905716), rotation= Rotation(pitch=0.0, yaw=-89.99999237060547, roll=0.0)),
        #                                                                 Transform(location= Location(x=396.0, y=275.0, z=0.032674599438905716), rotation= Rotation(pitch=0.0, yaw=-89.99999237060547, roll=0.0)),
        #                                                                 Transform(location= Location(x=396.0, y=275.0, z=0.032674599438905716), rotation= Rotation(pitch=0.0, yaw=-89.99999237060547, roll=0.0)),
        #                                                                 Transform(location= Location(x=396.0, y=275.0, z=0.032674599438905716), rotation= Rotation(pitch=0.0, yaw=-89.99999237060547, roll=0.0)),
        #                                                                 Transform(location= Location(x=396.0, y=275.0, z=0.032674599438905716), rotation= Rotation(pitch=0.0, yaw=-89.99999237060547, roll=0.0)),
        #                                                                 Transform(location= Location(x=396.0, y=275.0, z=0.032674599438905716), rotation= Rotation(pitch=0.0, yaw=-89.99999237060547, roll=0.0)),
        #                                                                 Transform(location= Location(x=396.0, y=275.0, z=0.032674599438905716), rotation= Rotation(pitch=0.0, yaw=-89.99999237060547, roll=0.0)),
        #                                                                 Transform(location= Location(x=396.0, y=275.0, z=0.032674599438905716), rotation= Rotation(pitch=0.0, yaw=-89.99999237060547, roll=0.0)),
        #                                                                 Transform(location= Location(x=396.0, y=275.0, z=0.032674599438905716), rotation= Rotation(pitch=0.0, yaw=-89.99999237060547, roll=0.0))]))
                                                                        
        # send_msg(cloud_conn, pickle.dumps(planner_data))
        # print("Planner data sent")

        # control_data = ControlMessage(steer=1.2345667, 
        #                                 throttle=1.2345667, 
        #                                 brake=1.2345667,
        #                                 hand_brake=1.2345667, 
        #                                 reverse=1.2345667, 
        #                                 timestamp=1.2345667)
        # send_msg(cloud_conn, pickle.dumps(control_data))
        # print("Control data sent")

if __name__=='__main__':
    connect_to_server()
