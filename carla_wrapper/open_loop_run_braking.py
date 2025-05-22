import carla
import time
import socket
import pickle
import zlib
import numpy as np
import os
import cv2
import csv
from queue import Queue
from threading import Thread
from ol_visualizer import Visualizer
from utils.service import send_msg, recv_msg
import braking_params as params

os.environ["SDL_AUDIODRIVER"] = "dummy"

SERVER_HOST = params.server_host
SERVER_PORT = params.server_port

# ==== Experiment Parameters ====
VEHICLE_NAME = params.vehicle_name
VEHICLE_SPEED = params.vehicle_speed
OBJECT_TYPE = params.object_type
PLATFORM = params.platform
MODEL = params.model
ACTIVE_REP = params.active_rep

# if VEHICLE_NAME == "audi.tt":
#     if VEHICLE_SPEED == "20mph":
#         VEHICLE_SPEED_INT = "22mph"
#     elif VEHICLE_SPEED == "44mph":
#         VEHICLE_SPEED_INT = "40mph"
#     elif VEHICLE_SPEED == "60mph":
#         VEHICLE_SPEED_INT = "64mph"
#     else:
#         VEHICLE_SPEED_INT = VEHICLE_SPEED  # fallback
# elif VEHICLE_NAME == "carlamotors.carlacola":
#     if VEHICLE_SPEED == "20mph":
#         VEHICLE_SPEED_INT = "20mph"
#     elif VEHICLE_SPEED == "40mph":
#         VEHICLE_SPEED_INT = "44mph"
#     elif VEHICLE_SPEED == "60mph":
#         VEHICLE_SPEED_INT = "66mph"
#     else:
#         VEHICLE_SPEED_INT = VEHICLE_SPEED  # fallback
# else:
#     VEHICLE_SPEED_INT = VEHICLE_SPEED  # no change for bike

# ==== Derived Speed Parameters ====
if "kph" in VEHICLE_SPEED:
    TARGET_SPEED_MPS = float(VEHICLE_SPEED.replace("kph", "")) / 3.6
elif "mph" in VEHICLE_SPEED:
    TARGET_SPEED_MPS = float(VEHICLE_SPEED.replace("mph", "")) * 0.44704
else:
    raise ValueError("VEHICLE_SPEED must end with 'kph' or 'mph'")

# ==== Logging Setup ====
LOG_DIR = "final/obs_type"#f"all_yolo_braking_results_{MODEL}_{PLATFORM}"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = f"{MODEL}_{PLATFORM}_{VEHICLE_NAME}_{VEHICLE_SPEED}_{OBJECT_TYPE}_{ACTIVE_REP}_latency_location_log.csv"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILENAME)

class CameraStreamer:
    def __init__(self, client, world, save_frames=False):
        self.client = client
        self.world = world
        self.visualizer = Visualizer(world)
        self.rgb_queue = Queue(maxsize=10)
        self.command_queue = Queue()
        self.frame_counter = 0
        self.transmit = True
        self.save_frames = save_frames

        self.first_brake_time = None
        self.brake_applied_location = None
        self.stopped_time = None
        self.stopped_location = None
        self.brake_engaged = False

        self.start_time = None
        self.frame_latency_data = {}

        self.vehicle = self.spawn_vehicle()
        self.rgb_camera = self.spawn_camera()
        self.obstacle = self.find_obstacle()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(SERVER_HOST, SERVER_PORT)
        self.sock.connect((SERVER_HOST, SERVER_PORT))
        print("[Client] Connected to server")

        self.listener_thread = Thread(target=self.listen_for_commands, daemon=True)
        self.listener_thread.start()

        if self.save_frames:
            self.save_dir = "braking_exp_frames_no_annotations"
            os.makedirs(self.save_dir, exist_ok=True)

        self.latency_log_file = open(LOG_PATH, mode="w", newline="")
        self.latency_logger = csv.DictWriter(self.latency_log_file, fieldnames=[
            "frame_id",
            "frame_capture_time", "frame_capture_loc",
            "frame_send_time", "frame_send_loc",
            "server_processing_time",
            "server_response_receive_time", "server_response_loc",
            "brake_issued",
            "stopping_time", "stopping_loc",
            "vehicle_speed_mph"
        ])
        self.latency_logger.writeheader()

        self.last_speed = 0.0
        self.last_time = time.time()
        self.filtered_decel = 0.0
        self.brake_cmd = 0.0

        self.use_diff = False
        self.prev_frame = None
        self.keyframe_interval = 10
        
        self.set_speed_once = False
        self.speed_forced = False

    def find_obstacle(self):
        if OBJECT_TYPE == "car":
            candidates = self.world.get_actors().filter("vehicle.tesla.model3")
        elif OBJECT_TYPE == "person":
            candidates = self.world.get_actors().filter("walker.pedestrian.0001")
        elif OBJECT_TYPE == "bike":
            candidates = self.world.get_actors().filter("vehicle.kawasaki.ninja")
        
        if len(candidates) == 1:
            print(f"[Client] Obstacle found: ID={candidates[0].id}")
            return candidates[0]
        else:
            print("[ERROR][Client] No matching obstacle found")
            return None

    def pid_throttle(self, current_speed, target_speed, kp=0.6, deadband=1):
        error = target_speed - current_speed
        # if abs(error) > 5:
        #     return 1.0
        if abs(error) < deadband:
            return 0.4  # minimal throttle to maintain speed
        return max(0.0, min(kp * error, 1.0))

    def compute_filtered_decel(self, current_speed, current_time, alpha=0.3):
        dt = current_time - self.last_time
        raw_decel = (self.last_speed - current_speed) / dt if dt > 0 else 0.0
        raw_decel = max(0.0, raw_decel)
        self.filtered_decel = alpha * raw_decel + (1 - alpha) * self.filtered_decel
        self.last_speed = current_speed
        self.last_time = current_time
        return self.filtered_decel

    # def brake_controller(self, target_decel=7, max_decel=7.3, kp=0.2, max_ramp=0.05): # audi.TT
    # def brake_controller(self, target_decel=6, max_decel=6, kp=0.2, max_ramp=0.05): # bike
    # def brake_controller(self, target_decel=3, max_decel=3, kp=0.45, max_ramp=0.15): # carlacola
    def brake_controller(self, target_decel=params.target_decel, max_decel=params.max_decel, kp=params.kp, max_ramp=params.max_ramp):
        if self.filtered_decel >= max_decel:
            self.brake_cmd = max(0.0, self.brake_cmd - max_ramp)
            return self.brake_cmd

        error = target_decel - self.filtered_decel
        desired_brake = kp * error
        desired_brake = max(0.0, min(desired_brake, 1.0))

        if desired_brake > self.brake_cmd:
            self.brake_cmd = min(self.brake_cmd + max_ramp, desired_brake)
        else:
            self.brake_cmd = max(self.brake_cmd - max_ramp, desired_brake)

        return self.brake_cmd

    def spawn_vehicle(self):
        vehicle = self.world.get_actors().filter(f'vehicle.{VEHICLE_NAME}')[0]
        print(f"[Client] Vehicle '{VEHICLE_NAME}' spawned")
        return vehicle

    def spawn_camera(self):
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.visualizer.display_width))
        bp.set_attribute('image_size_y', str(self.visualizer.display_height))
        bp.set_attribute('fov', '60')
        bbox = self.vehicle.bounding_box.extent
        camera_transform = carla.Transform(
                carla.Location(x=2.5,y=0, z=1.8), #x=2.5
                carla.Rotation(pitch=-10))
        # if 'audi' in VEHICLE_NAME:
        #     # Camera placement for sedan (Audi TT)
        #     camera_transform = carla.Transform(
        #         carla.Location(x=1.5, z=1.8),
        #         carla.Rotation(pitch=-10))
        # elif 'carlacola' in VEHICLE_NAME:
        #     # Camera placement for cola truck
        #     camera_transform = carla.Transform(
        #         carla.Location(x=2.0, z=3.0),
        #         carla.Rotation(pitch=-10))
        # elif 'kawasaki.ninja' in VEHICLE_NAME:
        #     # Camera placement for bike
        #     camera_transform = carla.Transform(
        #         carla.Location(x=0.3, z=1.4),
        #         carla.Rotation(pitch=-5))
        camera = self.world.spawn_actor(bp, camera_transform, attach_to=self.vehicle)

        def safe_camera_callback(image):
            try:
                expected_len = image.height * image.width * 4
                if len(image.raw_data) != expected_len:
                    return
                self.rgb_queue.put_nowait((image, self.vehicle.get_location(), time.time()))
            except Exception as e:
                x = 12
                #print("[Camera callback error]:", e)

        camera.listen(safe_camera_callback)
        print("[Client] Camera attached and listening")
        return camera

    def listen_for_commands(self):
        while True:
            try:
                msg = recv_msg(self.sock)
                if msg is None:
                    break
                command = pickle.loads(zlib.decompress(msg))
                if command.get('brake', False) is True:
                    self.command_queue.put(command)
            except Exception as e:
                print("[Client] Command listener error:", e)
                break

    def run(self):
        try:
            throttle, steer, brake = 0.0, 0.0, 0.0
            self.start_time = time.time()

            #Timing for fps calculation
            last_time = time.time()
            frame_counter_window = 0

            while True:
                current_time = time.time()
                sim_time = current_time - self.start_time
                now = time.time()

                velocity = self.vehicle.get_velocity()
                speed_mps = np.linalg.norm([velocity.x, velocity.y, velocity.z])
                speed_mph = speed_mps * 2.23694

                if not self.brake_engaged:
                    current_loc = self.vehicle.get_location()

                    # One-time velocity setting when in the right region
                    if current_loc.y < 90 and not self.speed_forced:
                        self.set_speed_once = True
                        self.speed_forced = True

                    if self.set_speed_once:
                        # # Convert to m/s once at start
                        # if VEHICLE_SPEED.endswith("mph"):
                        #     speed_mph_target = float(VEHICLE_SPEED.replace("mph", ""))
                        #     TARGET_SPEED_MPS = speed_mph_target / 2.237
                        # else:
                        #     raise ValueError(f"Unrecognized speed format: {VEHICLE_SPEED}")
    #     if set_speed_to_60:
                #         v_mps = 60 / 2.237  # 60 mph in m/s
                #         current_yaw = self.vehicle.get_transform().rotation.yaw
                #         forward = carla.Rotation(yaw=current_yaw).get_forward_vector()
                #         velocity_vector = carla.Vector3D(forward.x * v_mps,
                #                                         forward.y * v_mps,
                #                                         forward.z * v_mps)
                #         self.vehicle.set_target_velocity(velocity_vector)
                        current_yaw = self.vehicle.get_transform().rotation.yaw
                        forward = carla.Rotation(yaw=current_yaw).get_forward_vector()
                        velocity_vector = carla.Vector3D(
                            forward.x * TARGET_SPEED_MPS,
                            forward.y * TARGET_SPEED_MPS,
                            forward.z * TARGET_SPEED_MPS
                        )
                        self.vehicle.set_target_velocity(velocity_vector)

                        # Give control to PID on next iteration
                        self.set_speed_once = False

                    else:
                        # Regular PID control
                        throttle = self.pid_throttle(speed_mps, TARGET_SPEED_MPS)
                        brake = 0.0


                if not self.command_queue.empty():
                    cmd = self.command_queue.get()
                    if isinstance(cmd, dict) and cmd.get('brake', True):
                        # throttle, brake = 0.0, 1.0
                        self.transmit = False
                        self.brake_engaged = True
                        response_time = sim_time
                        response_loc = self.vehicle.get_location()
                        if self.obstacle is not None:
                            self.obstacle.destroy()
                            print("[Client] Obstacle destroyed after brake command")
                            self.obstacle = None

                        frame_id = cmd.get('frame_id', None)
                        server_proc_time = cmd.get('server_processing_time', None)

                        if self.first_brake_time is None:
                            self.first_brake_time = sim_time
                            self.brake_applied_location = (
                                round(response_loc.x, 2), round(response_loc.y, 2), round(response_loc.z, 2)
                            )
                            print(f"[Client] BRAKE applied at time {self.first_brake_time:.2f}, position: {self.brake_applied_location}")

                            if frame_id in self.frame_latency_data:
                                self.frame_latency_data[frame_id]["brake_issued"] = True
                                self.frame_latency_data[frame_id]["server_processing_time"] = round(server_proc_time, 6)
                                self.frame_latency_data[frame_id]["server_response_receive_time"] = round(response_time, 6)
                                self.frame_latency_data[frame_id]["server_response_loc"] = self.brake_applied_location

                if self.brake_engaged:
                    current_speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
                    filtered_decel = self.compute_filtered_decel(current_speed, current_time)
                    brake = self.brake_controller()
                    throttle = 0.0
                    print(f"[Client][BRAKING] Time: {sim_time:.2f}s | Speed: {current_speed:.2f} m/s | Decel: {filtered_decel:.2f} m/s² | Brake: {brake:.2f}")

                if self.brake_engaged and self.stopped_time is None:
                    if speed_mps < 0.1:
                        self.stopped_time = sim_time
                        loc = self.vehicle.get_location()
                        self.stopped_location = (
                            round(loc.x, 2), round(loc.y, 2), round(loc.z, 2)
                        )
                        print(f"[Client] Vehicle came to full stop at time {self.stopped_time:.2f}, position: {self.stopped_location}")
                        for record in self.frame_latency_data.values():
                            if record["brake_issued"] and record["stopping_time"] is None:
                                record["stopping_time"] = round(self.stopped_time, 6)
                                record["stopping_loc"] = self.stopped_location
                        break

                if self.transmit and not self.rgb_queue.empty():
                    image, capture_loc, capture_time = self.rgb_queue.get()
                    try:
                        array = np.frombuffer(image.raw_data, dtype=np.uint8).copy()
                        array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
                        self.visualizer.rgb_array = array
                        
                        self.frame_latency_data[self.frame_counter] = {
                            "frame_id": self.frame_counter,
                            "frame_capture_time": round(capture_time, 6),
                            "frame_capture_loc": (round(capture_loc.x, 2), round(capture_loc.y, 2), round(capture_loc.z, 2)),
                            "server_processing_time": None,
                            "server_response_receive_time": None,
                            "server_response_loc": None,
                            "brake_issued": False,
                            "stopping_time": None,
                            "stopping_loc": None,
                            "vehicle_speed_mph": round(speed_mph, 2)
                        }

                        is_keyframe = (self.frame_counter % self.keyframe_interval == 0) or self.prev_frame is None

                        if self.use_diff and not is_keyframe and self.prev_frame is not None:
                            # D-frame: XOR and JPEG compress the delta
                            diff = cv2.bitwise_xor(array, self.prev_frame)
                            success, jpeg = cv2.imencode('.jpg', diff, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                            frame_type = 'D'
                        else:
                            # I-frame: full frame JPEG
                            # success, jpeg = cv2.imencode('.webp', array, [int(cv2.IMWRITE_WEBP_QUALITY), 70])
                            frame_type = 'I'

                        # if success:
                        #     compressed_bytes = jpeg.tobytes()

                        data = {
                            'frame_id': self.frame_counter,
                            'timestamp': capture_time,
                            'rgb_frame': array, #compressed_bytes,
                            'frame_type': frame_type,
                        }

                        # self.prev_frame = array.copy()

                        if self.save_frames:
                            frame_filename = os.path.join(self.save_dir, f"frame_{self.frame_counter:05d}.png")
                            bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(frame_filename, bgr)

                        serialized = zlib.compress(pickle.dumps(data))
                        #print("Compressed serialized size:", len(serialized), "bytes")
                        
                        # serialized = zlib.compress(pickle.dumps(data))
                        # print("Length of data: ", len(serialized))

                        send_time = time.time() - self.start_time
                        send_loc = self.vehicle.get_location()
                        self.frame_latency_data[self.frame_counter]["frame_send_time"] = round(send_time, 6),
                        self.frame_latency_data[self.frame_counter]["frame_send_loc"] = (round(send_loc.x, 2), round(send_loc.y, 2), round(send_loc.z, 2))

                        send_msg(self.sock, serialized)

                        # col_history = self.vehicle.get_collision_history()
                        control = self.vehicle.get_control()

                        # print(f"[Client] Sent frame {self.frame_counter} at {capture_time:.2f}s | pose: {send_loc} | speed: {speed_mph:.2f} mph | throttle {throttle: .2f}")
                        #print(f"control: {control}")
                        self.frame_counter += 1

                        #FPS Calculation
                        frame_counter_window += 1
                        if now - last_time >= 1.0:
                            #print(f"[Client] Effective FPS: {frame_counter_window/(now - last_time)}")
                            frame_counter_window = 0
                            last_time = now

                    except Exception as e:
                        print("[Client] Frame processing error:", e)

                # if "carlacola" in VEHICLE_NAME and VEHICLE_SPEED == "60mph" and not self.brake_engaged:
                #     set_speed_to_60 = False
                #     speed_forced = False
                #     current_loc = self.vehicle.get_location()
                #     if current_loc.y < 0 and not speed_forced:
                #         set_speed_to_60 = True
                #         speed_forced = True
                    
                #     if set_speed_to_60:
                #         v_mps = 60 / 2.237  # 60 mph in m/s
                #         current_yaw = self.vehicle.get_transform().rotation.yaw
                #         forward = carla.Rotation(yaw=current_yaw).get_forward_vector()
                #         velocity_vector = carla.Vector3D(forward.x * v_mps,
                #                                         forward.y * v_mps,
                #                                         forward.z * v_mps)
                #         self.vehicle.set_target_velocity(velocity_vector)

                # Logs to debug camera location
                vehicle_tf = self.vehicle.get_transform()
                camera_tf = self.rgb_camera.get_transform()
                relative_loc = camera_tf.location - vehicle_tf.location
                print(f"[VEHICLE] Location: {vehicle_tf.location}, Rotation: {vehicle_tf.rotation}")
                print(f"[CAMERA ] Location: {camera_tf.location}, Rotation: {camera_tf.rotation}")
                print(f"Camera is at relative offset: {relative_loc}")
                bbox = self.vehicle.bounding_box
                print(bbox.extent)

                self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
                self.visualizer.render(throttle, steer, brake, sim_time)
                # time.sleep(0.01)

        except KeyboardInterrupt:
            print("[Client] Interrupted")
        finally:
            self.cleanup()

    def cleanup(self):
        print("\n[Client] ==== FINAL REPORT ====")
        if self.first_brake_time is not None:
            print(f"[Client] Brake received at t={self.first_brake_time:.2f}, location: {self.brake_applied_location}")
        if self.stopped_time is not None:
            print(f"[Client] Vehicle stopped at t={self.stopped_time:.2f}, location: {self.stopped_location}")
        for record in self.frame_latency_data.values():
            self.latency_logger.writerow(record)
        self.latency_log_file.close()

        try:
            if self.rgb_camera:
                self.rgb_camera.destroy()
            if self.vehicle:
                self.vehicle.destroy()
            self.sock.close()
            self.visualizer.close()
        except:
            pass
        print(f"[Client] Cleaned up. Log saved at: {LOG_PATH}")

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    save_frames = False
    streamer = CameraStreamer(client, world, save_frames=save_frames)
    streamer.run()

if __name__ == "__main__":
    main()