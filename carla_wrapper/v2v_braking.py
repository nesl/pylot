import carla
import time
import math
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
import v2v_braking_params as params

os.environ["SDL_AUDIODRIVER"] = "dummy"

SERVER_HOST = params.server_host
SERVER_PORT = params.server_port

# ==== Experiment Parameters ====
VEHICLE_NAME = params.vehicle_name
VEHICLE_SPEED = params.vehicle_speed
OBJECT_TYPE = params.object_type
MODE = params.braking_mode
PLATFORM = params.platform
MODEL = params.model
ACTIVE_REP = params.active_rep
V2V_DELAY_S = params.v2v_delay_s
NW_DELAY_S    = params.nw_delay_s

# Camera FOV (use params if available, else default 60)
CAMERA_FOV_DEG = getattr(params, "camera_fov_deg", 60.0)

# ==== Derived Speed Parameters ====
if "kph" in VEHICLE_SPEED:
    TARGET_SPEED_MPS = float(VEHICLE_SPEED.replace("kph", "")) / 3.6
elif "mph" in VEHICLE_SPEED:
    TARGET_SPEED_MPS = float(VEHICLE_SPEED.replace("mph", "")) * 0.44704
else:
    raise ValueError("VEHICLE_SPEED must end with 'kph' or 'mph'")

# ==== Logging Setup ====
LOG_DIR = "v2v"
os.makedirs(LOG_DIR, exist_ok=True)
if MODE == "v2v":
    DELAY = V2V_DELAY_S
elif MODE == "perception" and PLATFORM == "cloud":
    DELAY = NW_DELAY_S
else:
    DELAY = 0
LOG_FILENAME = f"{MODE}_{MODEL}_{PLATFORM}_{VEHICLE_NAME}_{VEHICLE_SPEED}_{DELAY}_{ACTIVE_REP}_latency_location_log.csv"
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

        # Event/state
        self.first_brake_time = None           # time brake command received (V2)
        self.brake_applied_location = None     # V2 location at brake receipt
        self.stopped_time = None               # V2 stop time
        self.stopped_location = None           # V2 stop loc
        self.brake_engaged = False

        # Leader (V1) braking state
        self.v1_last_speed = 0.0
        self.v1_last_time = time.time()
        self.v1_filtered_decel = 0.0
        self.v1_brake_cmd = 0.0
        self.v1_brake_time = None              # when V1 starts braking
        self.v1_stop_time = None
        self.v1_stop_loc = None

        # Follower (V2) decel controller state
        self.last_speed = 0.0
        self.last_time = time.time()
        self.filtered_decel = 0.0
        self.brake_cmd = 0.0

        self.braking_mode = getattr(params, "braking_mode", "perception")

        self.use_diff = False
        self.prev_frame = None
        self.keyframe_interval = 10

        self.set_speed_once = False
        self.speed_forced = False

        self.start_time = None
        self.frame_latency_data = {}           # rows keyed by frame_id (int) and telemetry rows ("tele_N")

        # --- V2V shim (synthetic local “broadcast”) ---
        self.v2v_delay_s = getattr(params, "v2v_delay_s", 0.020)  # seconds; default 150 ms
        self.v2v_pending_time = None   # sim-time when the V2V should be considered "received"
        self.v2v_sent = False          # ensure only one V2V per episode
        self.v2v_emit_time = None      # when V1 began braking (emit)
        self.v2v_recv_time = None      # when V2 received (after delay)
        self.trigger_reason = None     # 'vision' or 'v2v' for logging


        # Actors
        self.vehicle = self.spawn_vehicle()
        self.rgb_camera = self.spawn_camera()
        self.obstacle = self.find_obstacle()
        print(self.obstacle)

        # Network
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(SERVER_HOST, SERVER_PORT)
        self.sock.connect((SERVER_HOST, SERVER_PORT))
        print("[Client] Connected to server")

        self.listener_thread = Thread(target=self.listen_for_commands, daemon=True)
        self.listener_thread.start()

        # Optional frame dump
        if self.save_frames:
            self.save_dir = "braking_exp_frames_no_annotations"
            os.makedirs(self.save_dir, exist_ok=True)

        # CSV writer with expanded schema
        self.latency_log_file = open(LOG_PATH, mode="w", newline="")
        self.latency_logger = csv.DictWriter(self.latency_log_file, fieldnames=[
            "row_type", "sim_time",
            "frame_id",
            "frame_capture_time", "frame_capture_loc",
            "frame_send_time", "frame_send_loc",
            "server_processing_time",
            "server_response_receive_time", "server_response_loc",
            "brake_issued",
            "trigger_reason", "v2v_emit_time", "v2v_recv_time",
            # Telemetry per row
            "v1_speed_mps", "v1_speed_mph",
            "v2_speed_mps", "v2_speed_mph",
            "v1_loc", "v2_loc",
            "gap_m",
            # Event timelines (copied/backfilled)
            "v1_brake_time", "v2_brake_time",
            "v1_stop_time", "v2_stop_time",
            "v1_stop_loc", "v2_stop_loc",
        ])
        self.latency_logger.writeheader()

    # ----------------- helpers & actor setup -----------------
    def find_obstacle(self):
        # Find a Tesla leader that's not our own ego car
        candidates = [a for a in self.world.get_actors().filter("vehicle.tesla.model3")]
        if not candidates:
            print("[ERROR][Client] No tesla.model3 found for V1")
            return None
        for a in candidates:
            if a.id != self.vehicle.id:
                print(f"[Client] Leader (V1) found: ID={a.id}")
                return a
        print("[ERROR][Client] Only ego tesla.model3 present; no separate V1 found")
        return None

    def pid_throttle(self, current_speed, target_speed, kp=0.6, deadband=1):
        error = target_speed - current_speed
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
        bp.set_attribute('fov', str(CAMERA_FOV_DEG))
        camera_transform = carla.Transform(
            carla.Location(x=2.5, y=0, z=1.8),
            carla.Rotation(pitch=-10)
        )
        camera = self.world.spawn_actor(bp, camera_transform, attach_to=self.vehicle)

        def safe_camera_callback(image):
            try:
                expected_len = image.height * image.width * 4
                if len(image.raw_data) != expected_len:
                    return
                self.rgb_queue.put_nowait((image, self.vehicle.get_location(), time.time()))
            except Exception:
                pass

        camera.listen(safe_camera_callback)
        print("[Client] Camera attached and listening")
        return camera

    def _set_speed_along_heading(self, actor, speed_mps: float):
        tf = actor.get_transform()
        curr_vel = actor.get_velocity()
        speed = math.sqrt(curr_vel.x**2 + curr_vel.y**2 + curr_vel.z**2)
        print("V1 Current speed:", speed)
        yaw = tf.rotation.yaw
        fwd = carla.Rotation(yaw=yaw).get_forward_vector()
        actor.set_target_velocity(carla.Vector3D(fwd.x * speed_mps, fwd.y * speed_mps, fwd.z * speed_mps))

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

    # ----------------- logging helpers -----------------
    def _telemetry_fields(self):
        """Return instantaneous V1/V2 speeds/poses and gap."""
        if self.obstacle is not None:
            v1_vel = self.obstacle.get_velocity()
            v1_speed_mps = float(np.linalg.norm([v1_vel.x, v1_vel.y, v1_vel.z]))
            v1_speed_mph = v1_speed_mps * 2.23694
            v1_pos = self.obstacle.get_location()
            v1_loc_tuple = (round(v1_pos.x, 2), round(v1_pos.y, 2), round(v1_pos.z, 2))
        else:
            v1_speed_mps = v1_speed_mph = 0.0
            v1_pos = None
            v1_loc_tuple = (None, None, None)

        v2_pos = self.vehicle.get_location()
        v2_loc_tuple = (round(v2_pos.x, 2), round(v2_pos.y, 2), round(v2_pos.z, 2))

        gap_m = round((v1_pos.y - v2_pos.y), 2) if v1_pos is not None else None
        print("Headway gap: ", gap_m)
        return v1_speed_mps, v1_speed_mph, v1_loc_tuple, v2_loc_tuple, gap_m

    def log_tick(self, sim_time, v2_speed_mps, v2_speed_mph):
        v1_speed_mps, v1_speed_mph, v1_loc_tuple, v2_loc_tuple, gap_m = self._telemetry_fields()
        self.frame_latency_data[f"tele_{len(self.frame_latency_data)}"] = {
            "row_type": "telemetry",
            "sim_time": round(sim_time, 6),
            "frame_id": None,
            "frame_capture_time": None,
            "frame_capture_loc": None,
            "frame_send_time": None,
            "frame_send_loc": None,
            "server_processing_time": None,
            "server_response_receive_time": None,
            "server_response_loc": None,
            "brake_issued": self.first_brake_time is not None,
            "trigger_reason": self.trigger_reason,
            "v2v_emit_time": None if self.v2v_emit_time is None else round(self.v2v_emit_time, 6),
            "v2v_recv_time": None if self.v2v_recv_time is None else round(self.v2v_recv_time, 6),
            "v1_speed_mps": round(v1_speed_mps, 3),
            "v1_speed_mph": round(v1_speed_mph, 2),
            "v2_speed_mps": round(v2_speed_mps, 3),
            "v2_speed_mph": round(v2_speed_mph, 2),
            "v1_loc": v1_loc_tuple,
            "v2_loc": v2_loc_tuple,
            "gap_m": gap_m,
            "v1_brake_time": None if self.v1_brake_time is None else round(self.v1_brake_time, 6),
            "v2_brake_time": None if self.first_brake_time is None else round(self.first_brake_time, 6),
            "v1_stop_time": None if self.v1_stop_time is None else round(self.v1_stop_time, 6),
            "v2_stop_time": None if self.stopped_time is None else round(self.stopped_time, 6),
            "v1_stop_loc": self.v1_stop_loc,
            "v2_stop_loc": self.stopped_location,
        }

    def backfill_brake(self):
        """Backfill brake only for rows at/after the event time."""
        if self.first_brake_time is None:
            return
        t = round(self.first_brake_time, 6)
        for rec in self.frame_latency_data.values():
            st = rec.get("sim_time")
            if st is None or st < t:
                continue
            rec["v2_brake_time"] = t
            rec["brake_issued"] = True

    def backfill_v2_stop(self):
        """Backfill V2 stop only for rows at/after the stop time."""
        if self.stopped_time is None or self.stopped_location is None:
            return
        t = round(self.stopped_time, 6)
        loc = self.stopped_location
        for rec in self.frame_latency_data.values():
            st = rec.get("sim_time")
            if st is None or st < t:
                continue
            rec["v2_stop_time"] = t
            rec["v2_stop_loc"] = loc


    def backfill_v1_stop(self):
        """Backfill V1 stop only for rows at/after the stop time."""
        if self.v1_stop_time is None or self.v1_stop_loc is None:
            return
        t = round(self.v1_stop_time, 6)
        loc = self.v1_stop_loc
        for rec in self.frame_latency_data.values():
            st = rec.get("sim_time")
            if st is None or st < t:
                continue
            rec["v1_stop_time"] = t
            rec["v1_stop_loc"] = loc


    # ----------------- main loop -----------------
    def run(self):
        try:
            throttle, steer, brake = 0.0, 0.0, 0.0
            self.start_time = time.time()

            last_time = time.time()
            frame_counter_window = 0

            # Run until both cars stopped (or timeout after brake)
            post_brake_timeout_s = 6.0  # safety net

            while True:
                current_time = time.time()
                sim_time = current_time - self.start_time
                now = current_time

                # --- Local V2V shim delivery ---
                if (self.braking_mode == "v2v"
                        and self.v2v_pending_time is not None
                        and not self.brake_engaged
                        and sim_time >= self.v2v_pending_time
                        and not self.v2v_sent):
                    self.v2v_recv_time = sim_time
                    self.command_queue.put({'brake': True, 'reason': 'v2v'})
                    self.v2v_sent = True
                    print(f"[V2V] delivered at {self.v2v_recv_time:.3f}s")

                # Ego (V2) speed
                velocity = self.vehicle.get_velocity()
                speed_mps = float(np.linalg.norm([velocity.x, velocity.y, velocity.z]))
                speed_mph = speed_mps * 2.23694

                # Pre-brake motion control (V2)
                if not self.brake_engaged:
                    current_loc = self.vehicle.get_location()
                    if current_loc.y < 100 and not self.speed_forced:
                        self.set_speed_once = True
                        self.speed_forced = True
                    if self.set_speed_once:
                        current_yaw = self.vehicle.get_transform().rotation.yaw
                        forward = carla.Rotation(yaw=current_yaw).get_forward_vector()
                        velocity_vector = carla.Vector3D(forward.x * TARGET_SPEED_MPS,
                                                         forward.y * TARGET_SPEED_MPS,
                                                         forward.z * TARGET_SPEED_MPS)
                        self.vehicle.set_target_velocity(velocity_vector)
                        self.set_speed_once = False
                    else:
                        throttle = self.pid_throttle(speed_mps, TARGET_SPEED_MPS)
                        brake = 0.0

                # Handle brake command(s)
                if not self.command_queue.empty():
                    cmd = self.command_queue.get()
                    if isinstance(cmd, dict) and cmd.get('brake', True):
                        self.transmit = False
                        self.brake_engaged = True
                        response_time = sim_time
                        response_loc = self.vehicle.get_location()
                        frame_id = cmd.get('frame_id', None)
                        server_proc_time = cmd.get('server_processing_time', None)

                        if self.first_brake_time is None:
                            self.trigger_reason = cmd.get('reason', 'perception')  # default to 'vision' if server
                            self.first_brake_time = response_time
                            self.brake_applied_location = (
                                round(response_loc.x, 2), round(response_loc.y, 2), round(response_loc.z, 2)
                            )
                            print(f"[Client] BRAKE applied at time {self.first_brake_time:.2f}, position: {self.brake_applied_location}")

                            # Backfill brake instantly
                            self.backfill_brake()

                            # Create an explicit telemetry row at the brake instant
                            self.log_tick(sim_time, speed_mps, speed_mph)

                        # If the referenced image frame exists, enrich it; otherwise it's fine (we have telemetry)
                        if frame_id in self.frame_latency_data:
                            rec = self.frame_latency_data[frame_id]
                            rec["brake_issued"] = True
                            if server_proc_time is not None:
                                rec["server_processing_time"] = round(server_proc_time, 6)
                            rec["server_response_receive_time"] = round(response_time, 6)
                            rec["server_response_loc"] = self.brake_applied_location

                # V2 braking dynamics
                if self.brake_engaged:
                    current_speed = speed_mps
                    filtered_decel = self.compute_filtered_decel(current_speed, current_time)
                    brake = self.brake_controller()
                    throttle = 0.0
                    print(f"[Client][BRAKING] t={sim_time:.2f}s | V2 {current_speed:.2f} m/s | decel {filtered_decel:.2f} | brake {brake:.2f}")

                # V2 stop detection
                if self.brake_engaged and self.stopped_time is None and speed_mps < 0.1:
                    self.stopped_time = sim_time
                    loc = self.vehicle.get_location()
                    self.stopped_location = (round(loc.x, 2), round(loc.y, 2), round(loc.z, 2))
                    print(f"[Client] V2 full stop at {self.stopped_time:.2f}s, pos={self.stopped_location}")
                    self.backfill_v2_stop()

                # V1 control & stop detection
                if self.obstacle is not None:
                    v1_loc = self.obstacle.get_location()
                    v1_y = v1_loc.y
                    # Simple policy: move until threshold, then brake to zero
                    v1_speed_cmd = 8.94 if v1_y > -20.0 else 0.0
                    if v1_speed_cmd > 0.0:
                        self._set_speed_along_heading(self.obstacle, v1_speed_cmd)
                        self.v1_brake_cmd = 0.0
                    else:
                        if self.v1_brake_time is None:
                            self.v1_brake_time = sim_time
                            if self.braking_mode == "v2v" and self.v2v_pending_time is None:
                                self.v2v_emit_time = self.v1_brake_time
                                self.v2v_pending_time = self.v1_brake_time + self.v2v_delay_s
                                self.v2v_sent = False
                                print(f"[V2V] scheduled: emit={self.v2v_emit_time:.3f}s recv={self.v2v_pending_time:.3f}s")

                        # swap in leader state for filtered decel calc
                        prev_last_speed, prev_last_time = self.last_speed, self.last_time
                        prev_filtered, prev_brake = self.filtered_decel, self.brake_cmd
                        self.last_speed = self.v1_last_speed
                        self.last_time = self.v1_last_time
                        self.filtered_decel = self.v1_filtered_decel
                        self.brake_cmd = self.v1_brake_cmd
                        # measure leader speed
                        v = self.obstacle.get_velocity()
                        v1_speed_now = float(np.linalg.norm([v.x, v.y, v.z]))
                        print("V1 Current speed (braking):", v1_speed_now)
                        if self.v1_stop_time is None and v1_speed_now < 0.5:
                            self.v1_stop_time = sim_time
                            v1_loc2 = self.obstacle.get_location()
                            self.v1_stop_loc = (round(v1_loc2.x, 2), round(v1_loc2.y, 2), round(v1_loc2.z, 2))
                            print(f"[Client] V1 full stop at {self.v1_stop_time:.2f}s, pos={self.v1_stop_loc}")
                            self.backfill_v1_stop()
                        now_ts = time.time()
                        self.compute_filtered_decel(v1_speed_now, now_ts, alpha=0.35)
                        v1_brake = self.brake_controller(target_decel=7.0, max_decel=7.3, kp=0.2, max_ramp=0.05)
                        self.obstacle.apply_control(carla.VehicleControl(throttle=0.0, brake=float(v1_brake)))
                        # persist leader state
                        self.v1_last_speed = self.last_speed
                        self.v1_last_time = self.last_time
                        self.v1_filtered_decel = self.filtered_decel
                        self.v1_brake_cmd = self.brake_cmd
                        # restore follower state
                        self.last_speed = prev_last_speed
                        self.last_time = prev_last_time
                        self.filtered_decel = prev_filtered
                        self.brake_cmd = prev_brake

                # ---------------- image transmit path ----------------
                if self.transmit and not self.rgb_queue.empty():
                    image, capture_loc, capture_time = self.rgb_queue.get()
                    try:
                        array = np.frombuffer(image.raw_data, dtype=np.uint8).copy()
                        array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
                        self.visualizer.rgb_array = array

                        # Snapshot telemetry for row
                        v1_speed_mps, v1_speed_mph, v1_loc_tuple, v2_loc_tuple, gap_m = self._telemetry_fields()

                        # Create image row
                        self.frame_latency_data[self.frame_counter] = {
                            "row_type": "image",
                            "sim_time": round(sim_time, 6),
                            "frame_id": self.frame_counter,
                            "frame_capture_time": round(capture_time, 6),
                            "frame_capture_loc": (round(capture_loc.x, 2), round(capture_loc.y, 2), round(capture_loc.z, 2)),
                            "server_processing_time": None,
                            "server_response_receive_time": None,
                            "server_response_loc": None,
                            "brake_issued": self.first_brake_time is not None,
                            "trigger_reason": self.trigger_reason,
                            "v2v_emit_time": None if self.v2v_emit_time is None else round(self.v2v_emit_time, 6),
                            "v2v_recv_time": None if self.v2v_recv_time is None else round(self.v2v_recv_time, 6),
                            # telemetry
                            "v1_speed_mps": round(v1_speed_mps, 3),
                            "v1_speed_mph": round(v1_speed_mph, 2),
                            "v2_speed_mps": round(speed_mps, 3),
                            "v2_speed_mph": round(speed_mph, 2),
                            "v1_loc": v1_loc_tuple,
                            "v2_loc": v2_loc_tuple,
                            "gap_m": gap_m,
                            # event snapshots (filled/backfilled as known)
                            "v1_brake_time": None if self.v1_brake_time is None else round(self.v1_brake_time, 6),
                            "v2_brake_time": None if self.first_brake_time is None else round(self.first_brake_time, 6),
                            "v1_stop_time": None if self.v1_stop_time is None else round(self.v1_stop_time, 6),
                            "v2_stop_time": None if self.stopped_time is None else round(self.stopped_time, 6),
                            "frame_send_time": None,
                            "frame_send_loc": None,
                            "v1_stop_loc": self.v1_stop_loc,
                            "v2_stop_loc": self.stopped_location,
                        }

                        # (optional) keyframe/delta encode marker if you use it downstream
                        is_keyframe = (self.frame_counter % self.keyframe_interval == 0) or self.prev_frame is None
                        frame_type = 'I'
                        if self.use_diff and not is_keyframe and self.prev_frame is not None:
                            diff = cv2.bitwise_xor(array, self.prev_frame)
                            success, jpeg = cv2.imencode('.jpg', diff, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                            frame_type = 'D'
                        # attach extra payload metadata for server (unchanged)
                        vehicle_tf = self.vehicle.get_transform()
                        ego_loc = vehicle_tf.location
                        ego_yaw_deg = vehicle_tf.rotation.yaw
                        W, H = self.visualizer.display_width, self.visualizer.display_height
                        data = {
                            'frame_id': self.frame_counter,
                            'timestamp': capture_time,
                            'rgb_frame': array,
                            'frame_type': frame_type,
                            'ego_pose': (round(ego_loc.x, 2), round(ego_loc.y, 2), round(ego_loc.z, 2), round(ego_yaw_deg, 2)),
                            'ego_speed_mps': float(speed_mps),
                            'image_size': (W, H),
                            'camera_fov': float(CAMERA_FOV_DEG),
                        }
                        if self.save_frames:
                            frame_filename = os.path.join(self.save_dir, f"frame_{self.frame_counter:05d}.png")
                            bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(frame_filename, bgr)

                        serialized = zlib.compress(pickle.dumps(data))
                        send_time = time.time() - self.start_time
                        send_loc = self.vehicle.get_location()
                        # NOTE: no stray comma — keep as float
                        self.frame_latency_data[self.frame_counter]["frame_send_time"] = round(send_time, 6)
                        self.frame_latency_data[self.frame_counter]["frame_send_loc"] = (round(send_loc.x, 2), round(send_loc.y, 2), round(send_loc.z, 2))
                        send_msg(self.sock, serialized)

                        self.frame_counter += 1
                        frame_counter_window += 1
                        if now - last_time >= 1.0:
                            frame_counter_window = 0
                            last_time = now

                    except Exception as e:
                        print("[Client] Frame processing error:", e)

                # Always write a telemetry tick (keeps timeline dense even when transmit=False)
                self.log_tick(sim_time, speed_mps, speed_mph)

                # Apply control & render
                self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
                self.visualizer.render(throttle, steer, brake, sim_time)

                # Exit criteria: after brake, both cars stopped OR timeout
                if self.first_brake_time is not None:
                    timed_out = (sim_time - self.first_brake_time) > post_brake_timeout_s
                    if (self.stopped_time is not None) and (self.v1_stop_time is not None):
                        break
                    if timed_out and (self.stopped_time is not None):
                        # Give up on V1 stop after timeout, but we've logged V2 fully
                        break

        except KeyboardInterrupt:
            print("[Client] Interrupted")
        finally:
            self.cleanup()

    # ----------------- cleanup -----------------
    def cleanup(self):
        print("\n[Client] ==== FINAL REPORT ====")
        if self.first_brake_time is not None:
            print(f"[Client] Brake received at t={self.first_brake_time:.2f}, location: {self.brake_applied_location}")
        if self.stopped_time is not None:
            print(f"[Client] V2 stopped at t={self.stopped_time:.2f}, location: {self.stopped_location}")
        if self.v1_stop_time is not None:
            print(f"[Client] V1 stopped at t={self.v1_stop_time:.2f}, location: {self.v1_stop_loc}")

        # One last backfill sweep (idempotent)
        self.backfill_brake()
        self.backfill_v2_stop()
        self.backfill_v1_stop()

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
