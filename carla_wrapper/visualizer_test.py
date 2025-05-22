import carla
import pygame
import numpy as np
import time
import os

from ol_visualizer import Visualizer

# Prevent audio conflicts
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["SDL_VIDEODRIVER"] = "x11"

class VisualizerTest:
    def __init__(self, client, world):
        self.client = client
        self.world = world
        self.visualizer = Visualizer(world)
        self.rgb_surface = None

        self.vehicle = self.spawn_vehicle()
        self.rgb_camera = self.spawn_camera()

    def spawn_vehicle(self):
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_point = self.world.get_map().get_spawn_points()[0]
        vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True)
        print("[Test] Vehicle spawned and set to autopilot")
        return vehicle

    def spawn_camera(self):
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.visualizer.display_width))
        bp.set_attribute('image_size_y', str(self.visualizer.display_height))
        bp.set_attribute('fov', '90')
        bp.set_attribute('sensor_tick', '0.1')
        transform = carla.Transform(carla.Location(x=1.5, z=1.8), carla.Rotation(pitch=-10))
        camera = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        self.visualizer.set_camera_listener(camera, camera_type='rgb')
        print("[Test] Camera spawned and attached")
        return camera

    def run(self):
        try:
            throttle, steer, brake = 0.0, 0.0, 0.0
            start = time.time()

            while True:
                self.visualizer.render(throttle, steer, brake, time.time() - start)
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("[Test] Interrupted")
        finally:
            if self.rgb_camera:
                self.rgb_camera.destroy()
            if self.vehicle:
                self.vehicle.destroy()
            self.visualizer.close()
            print("[Test] Cleaned up")

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    test = VisualizerTest(client, world)
    test.run()

if __name__ == "__main__":
    main()
