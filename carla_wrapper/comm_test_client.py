import carla
import time
import socket
import pickle
import numpy as np
import pygame
import sys
import zlib
from ol_visualizer import Visualizer
from utils.service import send_msg, recv_msg

class CameraMonitorPipeline:
    def __init__(self, client, world, server_host='10.0.0.6', server_port=10000):
        self.world = world
        self.visualizer = Visualizer(world)
        self.rgb_camera, self.depth_camera = self.spawn_cameras()
        
        # Set up connection to the server
        self.server_host = server_host
        self.server_port = server_port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_host, self.server_port))

    def spawn_cameras(self):
        camera_transform = carla.Transform(
            carla.Location(x=200.0, y=303.0, z=5.0), carla.Rotation(pitch=-30, yaw=180))

        # RGB Camera
        rgb_camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_camera_bp.set_attribute('image_size_x', '640')
        rgb_camera_bp.set_attribute('image_size_y', '640')
        rgb_camera_bp.set_attribute('fov', '90')
        rgb_camera = self.world.spawn_actor(rgb_camera_bp, camera_transform)
        self.visualizer.set_camera_listener(rgb_camera, camera_type='rgb')

        # Depth Camera
        depth_camera_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        depth_camera_bp.set_attribute('image_size_x', '640')
        depth_camera_bp.set_attribute('image_size_y', '640')
        depth_camera_bp.set_attribute('fov', '90')
        depth_camera = self.world.spawn_actor(depth_camera_bp, camera_transform)
        self.visualizer.set_camera_listener(depth_camera, camera_type='depth')

        return rgb_camera, depth_camera

    def capture_frame(self):
        # Capture RGB and depth frames as numpy arrays
        rgb_frame = np.array(pygame.surfarray.array3d(self.visualizer.rgb_surface).swapaxes(0, 1)) if self.visualizer.rgb_surface else None
        depth_frame = np.array(pygame.surfarray.array3d(self.visualizer.depth_surface).swapaxes(0, 1)) if self.visualizer.depth_surface else None
        
        # Capture the camera's position and rotation (pose)
        camera_transform = self.rgb_camera.get_transform()
        pose = {
            'position': (camera_transform.location.x, camera_transform.location.y, camera_transform.location.z),
            'rotation': (camera_transform.rotation.pitch, camera_transform.rotation.yaw, camera_transform.rotation.roll)
        }
        
        return rgb_frame, depth_frame, pose

    def send_frame_to_server(self):
        timestamp_start = time.time()
        rgb_frame, depth_frame, pose = self.capture_frame()

        # Prepare data packet with timestamp and all sensor data
        data = {
            'timestamp_send': timestamp_start,
            'rgb_frame': rgb_frame,
            'depth_frame': depth_frame,
            'pose': pose
        }
        
        serialized_data = pickle.dumps(data)
        size = sys.getsizeof(serialized_data)
        print(f"Size of pickled object: {size} bytes")
        compressed_data = zlib.compress(serialized_data)
        size = sys.getsizeof(compressed_data)
        print(f"Size of compressed object: {size} bytes")
        timestamp_send = time.time()
        send_msg(self.client_socket, compressed_data)

        # Receive response with server timestamp
        response_data = recv_msg(self.client_socket)
        timestamp_recv = time.time()  # Receive timestamp
        response = pickle.loads(response_data)
        print("response: ", response)

        # Calculate round-trip latency
        server_timestamp = response['server_timestamp']
        round_trip_time = timestamp_recv - timestamp_send
        #server_processing_time = server_timestamp - timestamp_send
        communication_latency = round_trip_time - server_timestamp

        print(f"Total Round-trip time: {round_trip_time:.4f} seconds")
        #print(f"Server processing time: {server_processing_time:.4f} seconds")
        print(f"Communication latency: {communication_latency:.4f} seconds")

    def run(self):
        try:
            while True:
                self.send_frame_to_server()
                #time.sleep(1)  # Adjust frequency as needed
        except KeyboardInterrupt:
            print("Client stopped.")
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
