import carla
import time
import socket
import pickle
import pygame
import numpy as np
import params
import zlib
from ol_visualizer import Visualizer
from utils.service import send_msg, recv_msg

localhost = '0.0.0.0'
rpi = '10.0.0.4'
jetson = '10.0.0.6'

class CameraMonitorPipeline:
    def __init__(self, client, world, server_host=localhost, server_port=10000):
    #def __init__(self, client, world, server_host=rpi, server_port=10000):
    #def __init__(self, client, world, server_host=jetson, server_port=10000):
        self.world = world
        self.visualizer = Visualizer(world)
        self.rgb_camera, self.depth_camera = self.spawn_cameras()

        print(self.visualizer.display_width, self.visualizer.display_height)

        # Set up connection to the server
        self.server_host = server_host
        self.server_port = server_port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_host, self.server_port))

        self.frame_count = 0
        
    def get_car_location_velocity(self):
        cars = self.world.get_actors().filter('vehicle.*')
        if len(cars) > 0:
            return cars[0].get_location(), cars[0].get_velocity()
        return carla.Location(x=0.0,y=0.0,z=0.0), carla.Vector3D(x=0.0,y=0.0,z=0.0)

    def spawn_cameras(self):
        # Initialize RGB and depth cameras
        camera_transform = carla.Transform(
            carla.Location(x=200.0, y=303.0, z=5.0), carla.Rotation(pitch=-30, yaw=180))

        # RGB Camera
        rgb_camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_camera_bp.set_attribute('image_size_x', str(self.visualizer.display_width))
        rgb_camera_bp.set_attribute('image_size_y', str(self.visualizer.display_height))
        rgb_camera_bp.set_attribute('fov', '90')
        rgb_camera = self.world.spawn_actor(rgb_camera_bp, camera_transform)
        self.visualizer.set_camera_listener(rgb_camera, camera_type='rgb')

        # Depth Camera
        depth_camera_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        depth_camera_bp.set_attribute('image_size_x', str(self.visualizer.display_width))
        depth_camera_bp.set_attribute('image_size_y', str(self.visualizer.display_height))
        depth_camera_bp.set_attribute('fov', '90')
        depth_camera = self.world.spawn_actor(depth_camera_bp, camera_transform)
        self.visualizer.set_camera_listener(depth_camera, camera_type='depth')

        return rgb_camera, depth_camera

    def send_frame_to_server(self, rgb_frame, depth_frame):
        if rgb_frame is not None and depth_frame is not None:
            self.frame_count = self.frame_count + 1
            # Prepare data with timestamp and frames
            camera_transform = self.rgb_camera.get_transform()
            car_location, car_velocity = self.get_car_location_velocity()
            data = {
                'frame': self.frame_count,
                'timestamp': time.time(),
                'rgb_frame': rgb_frame,
                'depth_frame': depth_frame,
                'pose': {
                    'position': (camera_transform.location.x, camera_transform.location.y, camera_transform.location.z),
                    'rotation': (camera_transform.rotation.pitch, camera_transform.rotation.yaw, camera_transform.rotation.roll)
                },
                'car_location': (round(car_location.x, 2), round(car_location.y, 2), round(car_location.z, 2)),
                'car_velocity': (round(car_velocity.x, 2), round(car_velocity.y, 2), round(car_velocity.z, 2)),
            }
            #print("data: ", data)

            # Serialize and send the data to the server
            serialized_data = zlib.compress(pickle.dumps(data))
            #print("Sending data: ", data)
            send_msg(self.client_socket, serialized_data)

    def run(self):
        try:
            first_time = time.time()
            while True:
                #time.sleep(0.05)  # Simulate a frame rate
                # Retrieve RGB and depth frames from the visualizer
                rgb_frame = np.array(pygame.surfarray.array3d(self.visualizer.rgb_surface).swapaxes(0, 1)) if self.visualizer.rgb_surface else None
                depth_frame = np.array(pygame.surfarray.array3d(self.visualizer.depth_surface).swapaxes(0, 1)) if self.visualizer.depth_surface else None

                # print(rgb_frame)
                # Send frames to the server
                start_time = time.time()
                self.send_frame_to_server(rgb_frame, depth_frame)
                #message = recv_msg(self.client_socket)
                time_taken = time.time() - start_time

                print(f"Time taken = {time_taken}")

                # Example of dummy control values; replace as needed
                throttle, steer, brake = 0.5, 0.0, 0.0

                # Render visualizer output with control values and timestamp
                self.visualizer.render(throttle, steer, brake, time.time()-first_time)
        except KeyboardInterrupt:
            print("Pipeline stopped.")
        finally:
            if self.rgb_camera:
                self.rgb_camera.destroy()
            if self.depth_camera:
                self.depth_camera.destroy()
            self.visualizer.close()
            self.client_socket.close()

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    pipeline = CameraMonitorPipeline(client, world)
    pipeline.run()

if __name__ == "__main__":
    main()
