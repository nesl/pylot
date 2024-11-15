import pandas as pd
import time
import numpy as np
from collections import deque
from control.controller import Controller

class Vector3D(object):
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z

class Location(Vector3D):
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        super().__init__(x, y, z)

class Rotation(object):
    def __init__(self, pitch: float = 0, yaw: float = 0, roll: float = 0):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

class Transform(object):
    def __init__(self, location: Location = None, rotation: Rotation = None, matrix=None):
        self.location = location
        self.rotation = rotation
        self.matrix = matrix

class Pose(object):
    def __init__(self, transform: Transform, forward_speed: float, velocity_vector: Vector3D = None, localization_time: float = None):
        self.transform = transform
        self.forward_speed = forward_speed
        self.velocity_vector = velocity_vector
        self.localization_time = localization_time

class Waypoints(object):
    def __init__(self, waypoints: deque, target_speeds: deque):
        self.waypoints = waypoints
        self.target_speeds = target_speeds

    @classmethod
    def read_from_csv_file(cls, csv_file_name: str, target_speed):
        csv_file = open(csv_file_name)
        csv_reader = csv.reader(csv_file)
        waypoints = []
        for row in csv_reader:
            x = float(row[0])
            y = float(row[1])
            z = float(row[2])
            waypoint = Transform(Location(x, y, z), Rotation(0, 0, 0))
            waypoints.append(waypoint)
        target_speeds = deque([target_speed for _ in range(len(waypoints))])
        return cls(deque(waypoints), target_speeds)

    def as_numpy_array_2D(self):
        wx = []
        wy =[]
        for wp in self.waypoints:
            wx.append(wp.location.x)
            wy.append(wp.location.y)
        return np.array([wx,wy])

class MPCRunner():

    def __init__(self):
        self._controller = Controller()
        self.throttle = -1
        self.brake = -1
        self.steer = -1

    def run_MPC(self, timestamp, pose, waypoints):
        start_time = time.time()
        #print(waypoints.waypoints)
        (steer, throttle, brake, controller_runtime) = self._controller.get_control_instructions(timestamp, pose, waypoints)
        end_time = time.time()
        print("\nControl instructions {} {} {} {}".format(throttle, steer, brake, controller_runtime))
        print("Calculated runtime: ", end_time - start_time)
        print("Forwarded runtime: ", controller_runtime)

        if throttle == 0 and brake == 0.5 and steer == 0:
            if self.throttle != -1 and self.brake != -1:
                throttle = self.throttle
                brake = self.brake
                steer = self.steer

        self.throttle = throttle
        self.brake = brake
        self.steer = steer

def parse_pose(pose_str):
    #print(pose_str)
    transform_str = pose_str.split('transform: ')[1].split('), forward_speed')[0]
    #print(transform_str)
    forward_speed = float(pose_str.split('forward speed: ')[1].split(',')[0])
    #print("forward speed : ", forward_speed)
    location_str = transform_str.split('location: Location(')[1].split('), rotation:')[0]
    rotation_str = transform_str.split('rotation: Rotation(')[1].split(')')[0]

    x, y, z = [float(value.split('=')[1]) for value in location_str.split(', ')]
    pitch, yaw, roll = [float(value.split('=')[1]) for value in rotation_str.split(', ')]

    location = Location(x, y, z)
    rotation = Rotation(pitch, yaw, roll)
    transform = Transform(location, rotation)

    return Pose(transform, forward_speed)

def parse_waypoints(waypoints_str):
    #print(waypoints_str)
    waypoints_list = waypoints_str[6:-1].split('), Transform')
    waypoints = []
    for waypoint_str in waypoints_list:
        waypoint_str = 'Transform' + waypoint_str if not waypoint_str.startswith('Transform') else waypoint_str
        #print(waypoint_str)
        location_str = waypoint_str.split('location: Location(')[1].split('), rotation:')[0]
        #print(location_str)
        rotation_str = waypoint_str.split('rotation: Rotation(')[1].split(')')[0]

        x, y, z = [float(value.split('=')[1]) for value in location_str.split(', ')]
        pitch, yaw, roll = [float(value.split('=')[1]) for value in rotation_str.split(', ')]

        location = Location(x, y, z)
        rotation = Rotation(pitch, yaw, roll)
        transform = Transform(location, rotation)

        waypoints.append(transform)

    target_speeds = deque([0] * len(waypoints))
    return Waypoints(deque(waypoints), target_speeds)

def main():
    file_path = 'planner_dump.csv'
    data = pd.read_csv(file_path)
    
    runner = MPCRunner()
    
    for index, row in data.iterrows():
        timestamp = row['timestamp']
        pose = parse_pose(row['pose'])
        waypoints = parse_waypoints(row['waypoints'])
        
        runner.run_MPC(timestamp, pose, waypoints)

if __name__ == '__main__':
    main()
