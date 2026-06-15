import os
os.environ["SDL_AUDIODRIVER"] = "dummy"

import time
import math
import socket
import pickle
import zlib
from queue import Queue
from threading import Thread
from contextlib import suppress

import numpy as np
import cv2
import carla

from ol_visualizer import Visualizer
from utils.service import send_msg, recv_msg
import v2i_braking_params as params

# -----------------------
# Derived/Unpacked Params
# -----------------------
SERVER_HOST = params.server_host
SERVER_PORT = params.server_port

VEHICLE_NAME  = params.vehicle_name
VEHICLE_SPEED = params.vehicle_speed

EGO_FOV_DEG = float(getattr(params, "ego_camera_fov_deg", 60.0))

INFRA_ENABLED    = bool(getattr(params, "infra_enabled", True))
INFRA_SIZE       = tuple(getattr(params, "infra_image_size", (1280, 720)))
INFRA_FOV_DEG    = float(getattr(params, "infra_fov_deg", 70.0))
INFRA_BASE_X     = float(getattr(params, "infra_base_x", -10.0))
INFRA_BASE_Y     = float(getattr(params, "infra_base_y", 28.0))
INFRA_BASE_Z     = float(getattr(params, "infra_base_z", 0.0))
INFRA_HEIGHT_M   = float(getattr(params, "infra_height_m", 8.0))
INFRA_PITCH_DEG  = float(getattr(params, "infra_pitch_deg", -15.0))
INFRA_YAW_DEG    = float(getattr(params, "infra_yaw_deg", 35.0))

if "kph" in VEHICLE_SPEED:
    TARGET_SPEED_MPS = float(VEHICLE_SPEED.replace("kph", "")) / 3.6
elif "mph" in VEHICLE_SPEED:
    TARGET_SPEED_MPS = float(VEHICLE_SPEED.replace("mph", "")) * 0.44704
else:
    raise ValueError("VEHICLE_SPEED must end with 'kph' or 'mph'")

TARGET_DECEL = float(getattr(params, "target_decel", 7.0))
MAX_DECEL    = float(getattr(params, "max_decel", 7.3))
KP           = float(getattr(params, "kp", 0.2))
MAX_RAMP     = float(getattr(params, "max_ramp", 0.05))

LOG_DIR = "v2i"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = f"{params.mode}_{params.occlusion}_{params.model}_{params.platform}_{VEHICLE_NAME}_{VEHICLE_SPEED}_person_{getattr(params, 'active_rep', 1)}_latency_location_log.csv"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILENAME)

class V2IClient:
    def __init__(self, world: carla.World):
        self.world = world

        # === Actors ===
        self.ego = self._get_ego_vehicle()
        self.ego_cam = None
        self.infra_cam = None

        # === Visualization (single window) ===
        self.ego_viz = Visualizer(world)       # only one pygame window
        self.show_infra = INFRA_ENABLED

        # latest frames for composition
        self.ego_frame = None
        self.infra_frame = None

        # === Queues ===
        self.ego_q = Queue(maxsize=5)
        self.infra_q = Queue(maxsize=5) if INFRA_ENABLED else None
        self.cmd_q = Queue()

        # === Sockets ===
        self.ego_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.infra_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) if INFRA_ENABLED else None

        # === Control State ===
        self.start_time = time.time()
        self.brake_engaged = False
        self.first_brake_time = None
        self.brake_applied_location = None
        self.stopped_time = None
        self.stopped_location = None

        self.last_speed = 0.0
        self.last_time = time.time()
        self.filtered_decel = 0.0
        self.brake_cmd = 0.0

        self.tx_enabled = True  # allow turning off after brake if you like

        self.send_time = None
        self.recv_time = None

        # CSV logger
        import csv
        self.log_file = open(LOG_PATH, mode="w", newline="")
        self.logger = csv.DictWriter(self.log_file, fieldnames=[
        "row_type","sim_time","device_id","frame_id",
        "frame_capture_time","frame_capture_loc",
        "frame_send_time","frame_send_loc",
        "server_processing_time",
        "server_response_receive_time","server_response_loc",
        "brake_issued",
        "x_ego","y_ego","z_ego",
        "ego_speed_mps_capture","ego_speed_mph_capture",
        "ego_speed_mps_log","ego_speed_mph_log",
        "p_x","p_y","p_z","v_px","v_py","v_pz","ped_speed_mps",
        "ped_started","ped_start_time",
        "crossing_x","crossing_y","crossing_z"
        ])
        self.logger.writeheader()

        # === Walker control (client-driven jaywalk) ===
        self.ped = None                 # carla.Actor (walker)
        self._walker_thread = None
        self._walker_run = True
        # goal_x can be overridden in params as ped_goal_x; else use -140 by default
        self._walker_goal_x = float(getattr(params, "ped_goal_x", -140.0))
        self._walker_speed = float(getattr(params, "ped_speed_mps", 1.4))
        self._walker_t_start = float(getattr(params, "t_ped_start_s", 4.0))
        # probe to identify the ped ScenarioRunner spawned
        self._ped_probe = carla.Location(
            float(getattr(params, "ped_start_x", -100.0)),
            float(getattr(params, "ped_start_y",  -65.0)),
            float(getattr(params, "ped_start_z",    0.6)),
        )

        # --- GT ped state for evaluation-only logging (do NOT use for control) ---
        self.gt_ped_loc = None
        self.gt_ped_vel = None
        self._prev_ped_loc = None
        self._prev_ped_time = None

    # ---------------------------
    # Setup helpers
    # ---------------------------
    def _get_ego_vehicle(self) -> carla.Vehicle:
        matches = self.world.get_actors().filter(f"vehicle.{VEHICLE_NAME}")
        if not matches:
            raise RuntimeError(f"[Client] Ego 'vehicle.{VEHICLE_NAME}' not found. Did you launch the scenario?")
        ego = matches[0]
        ego.set_autopilot(False)
        return ego

    def spawn_ego_camera(self):
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.ego_viz.display_width))
        bp.set_attribute('image_size_y', str(self.ego_viz.display_height))
        bp.set_attribute('fov', str(EGO_FOV_DEG))
        cam_tf = carla.Transform(
            carla.Location(x=2.5, y=0.0, z=1.6),
            carla.Rotation(pitch=-10.0)
        )
        cam = self.world.spawn_actor(bp, cam_tf, attach_to=self.ego)

        def cb(image: carla.Image):
            try:
                if len(image.raw_data) != image.height * image.width * 4:
                    return
                self.ego_q.put_nowait((image, self.ego.get_location(), time.time()))
            except Exception:
                pass

        cam.listen(cb)
        self.ego_cam = cam
        print("[Client] Ego camera attached and listening.")

    def spawn_infra_camera(self):
        if not INFRA_ENABLED:
            return
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        W, H = INFRA_SIZE
        bp.set_attribute('image_size_x', str(W))
        bp.set_attribute('image_size_y', str(H))
        bp.set_attribute('fov', str(INFRA_FOV_DEG))
        tf = carla.Transform(
            carla.Location(x=INFRA_BASE_X, y=INFRA_BASE_Y, z=INFRA_BASE_Z + INFRA_HEIGHT_M),
            carla.Rotation(pitch=INFRA_PITCH_DEG, yaw=INFRA_YAW_DEG)
        )
        cam = self.world.spawn_actor(bp, tf)

        def cb(image: carla.Image):
            try:
                if len(image.raw_data) != image.height * image.width * 4:
                    return
                self.infra_q.put_nowait((image, tf.location, time.time()))
            except Exception:
                pass

        cam.listen(cb)
        self.infra_cam = cam
        print("[Client] Infrastructure camera spawned and listening.")

    # ---------------------------
    # Networking
    # ---------------------------
    def connect(self):
        print(f"[Client] Connecting (ego) to {SERVER_HOST}:{SERVER_PORT} ...")
        self.ego_sock.connect((SERVER_HOST, SERVER_PORT))
        print("[Client] Ego socket connected.")
        if self.infra_sock:
            print(f"[Client] Connecting (infra) to {SERVER_HOST}:{SERVER_PORT} ...")
            self.infra_sock.connect((SERVER_HOST, SERVER_PORT))
            print("[Client] Infra socket connected.")

        Thread(target=self._listen_for_cmds, daemon=True).start()
        Thread(target=self._ego_sender_loop, daemon=True).start()
        if self.infra_sock:
            Thread(target=self._infra_sender_loop, daemon=True).start()

    def _listen_for_cmds(self):
        while True:
            try:
                msg = recv_msg(self.ego_sock)
                if msg is None:
                    break
                command = pickle.loads(zlib.decompress(msg))
                if command.get('brake', False):
                    self.cmd_q.put(command)
            except Exception as e:
                print("[Client] Command listener error:", e)
                break

    # ---------------------------
    # Control helpers
    # ---------------------------
    def _pid_throttle(self, current_speed, target_speed, kp=0.6, deadband=1.0):
        err = target_speed - current_speed
        if abs(err) < deadband:
            return 0.4
        return max(0.0, min(kp * err, 1.0))

    def _filtered_decel(self, current_speed, now, alpha=0.3):
        dt = max(1e-3, now - self.last_time)
        raw = max(0.0, (self.last_speed - current_speed) / dt)
        self.filtered_decel = alpha * raw + (1 - alpha) * self.filtered_decel
        self.last_speed = current_speed
        self.last_time = now
        return self.filtered_decel

    def _brake_controller(self):
        if self.filtered_decel >= MAX_DECEL:
            self.brake_cmd = max(0.0, self.brake_cmd - MAX_RAMP)
            return self.brake_cmd
        err = TARGET_DECEL - self.filtered_decel
        desired = max(0.0, min(KP * err, 1.0))
        if desired > self.brake_cmd:
            self.brake_cmd = min(self.brake_cmd + MAX_RAMP, desired)
        else:
            self.brake_cmd = max(self.brake_cmd - MAX_RAMP, desired)
        return self.brake_cmd

        # ---------------------------
    # Walker helpers (client-controlled jaywalk)
    # ---------------------------
    def _bind_pedestrian(self):
        """Find the walker spawned by ScenarioRunner near ped_start and detach any AI controller."""
        walkers = self.world.get_actors().filter("walker.pedestrian.*")
        if not walkers:
            print("[Client][walker] No walkers in world.")
            return None
        ped = min(walkers, key=lambda a: a.get_location().distance(self._ped_probe))

        # Detach/destroy any AI controller parented to this walker so it won’t override our controls.
        for a in self.world.get_actors().filter("controller.ai.walker"):
            with suppress(Exception):
                if a.parent and a.parent.id == ped.id:
                    a.stop()
                    a.destroy()

        with suppress(Exception): ped.set_simulate_physics(True)
        with suppress(Exception): ped.set_target_velocity(carla.Vector3D(0,0,0))
        with suppress(Exception): ped.apply_control(carla.WalkerControl(direction=carla.Vector3D(0,0,0), speed=0.0))
        print(f"[Client][walker] Bound to walker id={ped.id} at {ped.get_location()}")
        return ped

    def _walker_loop(self, is_sync=True):
        """Wait until t_ped_start_s, then walk straight along -X each tick until goal_x, then hold."""
        def tick():
            # In sync runs (ScenarioRunner typical), wait for server ticks to align control with frames.
            return self.world.wait_for_tick() if is_sync else time.sleep(0.02)

        # Bind once
        self.ped = self._bind_pedestrian()
        if self.ped is None:
            return

        # Idle until start time
        wait_until = self._walker_t_start + 4.0
        while self._walker_run and (time.time() - self.start_time) < wait_until:
            tick()

        fwd = carla.Vector3D(-1.0, 0.0, 0.0)  # straight across (jaywalk)
        eps = 0.25

        # Walk until goal_x
        while self._walker_run:
            loc = self.ped.get_location()
            print("[Client] Walker's current location: ", loc)
            if loc.x <= (self._walker_goal_x + eps):
                break
            # Re-apply every tick so it never “sticks”
            with suppress(Exception):
                self.ped.apply_control(carla.WalkerControl(direction=fwd, speed=self._walker_speed))
            tick()

        # Hold pose cleanly for a couple of ticks (stabilizes screenshots/metrics)
        for _ in range(3):
            with suppress(Exception):
                self.ped.apply_control(carla.WalkerControl(direction=fwd, speed=0.0))
            tick()

    # ---------------------------
    # Sender loops (also stash frames for UI)
    # ---------------------------
    def _ego_sender_loop(self):
        frame_id = 0
        while True:
            if not self.tx_enabled:
                time.sleep(0.01); continue
            if self.ego_q.empty():
                time.sleep(0.001); continue

            image, capture_loc, capture_time = self.ego_q.get()
            try:
                arr = np.frombuffer(image.raw_data, dtype=np.uint8)
                arr = arr.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]  # BGRA->RGB
                self.ego_frame = arr  # <-- stash latest ego frame

                ego_tf = self.ego.get_transform()
                ego_loc = ego_tf.location
                ego_yaw = ego_tf.rotation.yaw
                vel = self.ego.get_velocity()
                speed_mps = float(np.linalg.norm([vel.x, vel.y, vel.z]))
                speed_mph = speed_mps * 2.23694

                data = {
                    'device_id': 'ego',
                    'type': 'frame',
                    'frame_id': frame_id,
                    'timestamp': capture_time,
                    'rgb_frame': arr,
                    'frame_type': 'I',
                    'ego_pose': (round(ego_loc.x, 2), round(ego_loc.y, 2), round(ego_loc.z, 2), round(ego_yaw, 2)),
                    'ego_speed_mps': float(speed_mps),
                    'image_size': (self.ego_viz.display_width, self.ego_viz.display_height),
                    'camera_fov': float(EGO_FOV_DEG),
                    'sim_time': time.time() - self.start_time,           # add sim clock to align with server
                    'gt_ped_loc': getattr(self, 'gt_ped_loc', None),     # GT ped pose for evaluation
                    'gt_ped_vel_mps': getattr(self, 'gt_ped_vel', None), # GT ped velocity (m/s) for evaluation
                }
                serialized = zlib.compress(pickle.dumps(data))
                send_msg(self.ego_sock, serialized)

                self._log_image_row(sim_time=time.time() - self.start_time,
                                    device_id="ego",
                                    frame_id=frame_id,
                                    capture_time=capture_time,
                                    capture_loc=(round(capture_loc.x,2), round(capture_loc.y,2), round(capture_loc.z,2)),
                                    send_loc=(round(ego_loc.x,2), round(ego_loc.y,2), round(ego_loc.z,2)),
                                    speed_mps=speed_mps)
                frame_id += 1
            except Exception as e:
                print("[Client][ego] send err:", e)

    def _infra_sender_loop(self):
        frame_id = 0
        while True:
            if not self.tx_enabled or not self.infra_q:
                time.sleep(0.01); continue
            if self.infra_q.empty():
                time.sleep(0.001); continue

            image, cam_loc, capture_time = self.infra_q.get()
            try:
                arr = np.frombuffer(image.raw_data, dtype=np.uint8)
                arr = arr.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
                self.infra_frame = arr  # <-- stash latest infra frame

                data = {
                    'device_id': 'infra',
                    'type': 'frame',
                    'frame_id': frame_id,
                    'timestamp': capture_time,
                    'rgb_frame': arr,
                    'frame_type': 'I',
                    'ego_pose': None,
                    'ego_speed_mps': None,
                    'image_size': INFRA_SIZE,
                    'camera_fov': float(INFRA_FOV_DEG),
                }
                serialized = zlib.compress(pickle.dumps(data))
                send_msg(self.infra_sock, serialized)

                self._log_image_row(sim_time=time.time() - self.start_time,
                                    device_id="infra",
                                    frame_id=frame_id,
                                    capture_time=capture_time,
                                    capture_loc=(round(cam_loc.x,2), round(cam_loc.y,2), round(cam_loc.z,2)),
                                    send_loc=(round(cam_loc.x,2), round(cam_loc.y,2), round(cam_loc.z,2)),
                                    speed_mps=None)
                frame_id += 1
            except Exception as e:
                print("[Client][infra] send err:", e)

    # ---------------------------
    # UI composition (one window)
    # ---------------------------
    def _compose_tiled_frame(self):
        """
        Returns an RGB frame to show: side-by-side [ego | infra] if both available,
        otherwise whichever is available. Resized to the Visualizer display size.
        """
        W, H = self.ego_viz.display_width, self.ego_viz.display_height

        # pick available frames
        ego = self.ego_frame
        infra = self.infra_frame if self.show_infra else None

        if ego is None and infra is None:
            return None
        if ego is None:
            # just infra, resized to fit
            out = cv2.resize(infra, (W, H))
            return out
        if infra is None:
            out = cv2.resize(ego, (W, H))
            return out

        # both exist: match heights, then hstack
        h = min(ego.shape[0], infra.shape[0])
        def resize_h(frame, target_h):
            fh = frame.shape[0]
            fw = frame.shape[1]
            new_w = int(round(fw * (target_h / float(fh))))
            return cv2.resize(frame, (new_w, target_h))

        ego_r = resize_h(ego, h)
        infra_r = resize_h(infra, h)

        tiled = np.hstack([ego_r, infra_r])

        # final resize to our display
        out = cv2.resize(tiled, (W, H))
        return out

    # ---------------------------
    # Main loop: motion + UI + BRAKE
    # ---------------------------
    def run(self):

        throttle, steer, brake = 0.0, 0.0, 0.0

        self.spawn_ego_camera()
        if INFRA_ENABLED:
            self.spawn_infra_camera()
        self.connect()

        # Kick off client-controlled walker thread (sync=True fits ScenarioRunner default)
        self._walker_thread = Thread(target=self._walker_loop, kwargs={"is_sync": True}, daemon=True)
        self._walker_thread.start()

        self.ped = self._bind_pedestrian()

        set_speed_once = True
        post_brake_timeout_s = 60.0

        try:
            while True:
                now = time.time()
                sim_time = now - self.start_time

                vel = self.ego.get_velocity()
                speed_mps = float(np.linalg.norm([vel.x, vel.y, vel.z]))
                # Pedestrian location (if spawned)
                ped_loc = None
                if self.ped:
                    try:
                        pl = self.ped.get_location()
                        ped_loc = (round(pl.x,2), round(pl.y,2), round(pl.z,2))
                    except RuntimeError:
                        ped_loc = None

                # --- Update GT ped state (pose + velocity) for evaluation/logging only ---
                if ped_loc is not None:
                    if self._prev_ped_loc is not None and self._prev_ped_time is not None:
                        dt = max(1e-3, sim_time - self._prev_ped_time)
                        vx = (ped_loc[0] - self._prev_ped_loc[0]) / dt
                        vy = (ped_loc[1] - self._prev_ped_loc[1]) / dt
                        vz = (ped_loc[2] - self._prev_ped_loc[2]) / dt
                        self.gt_ped_vel = (round(vx, 3), round(vy, 3), round(vz, 3))
                    self.gt_ped_loc = ped_loc
                    self._prev_ped_loc = ped_loc
                    self._prev_ped_time = sim_time


                # Ego location
                ego_loc = self.ego.get_location()
                ego_tuple = (round(ego_loc.x,2), round(ego_loc.y,2), round(ego_loc.z,2))

                # Update decel estimate
                self._filtered_decel(speed_mps, now, alpha=0.3)

                print(f"[Step] t={sim_time:5.2f}s | "
                    f"Ego@{ego_tuple} v={speed_mps:4.2f}m/s "
                    f"decel={self.filtered_decel:4.2f} "
                    f"brake={self.brake_engaged} | "
                    f"Ped@{ped_loc}")

                # Pre-brake motion control
                if not self.brake_engaged:
                    if set_speed_once:
                        yaw = self.ego.get_transform().rotation.yaw
                        fwd = carla.Rotation(yaw=yaw).get_forward_vector()
                        self.ego.set_target_velocity(carla.Vector3D(fwd.x * TARGET_SPEED_MPS,
                                                                    fwd.y * TARGET_SPEED_MPS,
                                                                    fwd.z * TARGET_SPEED_MPS))
                        set_speed_once = False
                    else:
                        throttle = self._pid_throttle(speed_mps, TARGET_SPEED_MPS)
                        brake = 0.0

                # BRAKE command
                if not self.cmd_q.empty():
                    cmd = self.cmd_q.get()
                    if isinstance(cmd, dict) and cmd.get('brake', True):
                        self.tx_enabled = False
                        self.brake_engaged = True
                        self.first_brake_time = sim_time
                        loc = self.ego.get_location()
                        self.brake_applied_location = (round(loc.x,2), round(loc.y,2), round(loc.z,2))
                        print(f"[Client] BRAKE at {self.first_brake_time:.2f}s, ego={self.brake_applied_location}")
                        self._log_cmd(sim_time, cmd, self.brake_applied_location)

                # Braking dynamics
                if self.brake_engaged:
                    self._filtered_decel(speed_mps, now, alpha=0.3)
                    brake = self._brake_controller()
                    throttle = 0.0
                    print(f"[BrakeEngaged] t={sim_time:5.2f}s | "
                            f"Ego@{self.brake_applied_location} | Ped@{ped_loc}")


                # Stop detection
                if self.brake_engaged and self.stopped_time is None and speed_mps < 0.1:
                    self.stopped_time = sim_time
                    loc = self.ego.get_location()
                    self.stopped_location = (round(loc.x,2), round(loc.y,2), round(loc.z,2))
                    print(f"[Client] Ego STOP at {self.stopped_time:.2f}s, ego={self.stopped_location}")
                    print(f"[Stopped] t={self.stopped_time:5.2f}s | "
                            f"Ego@stop {self.stopped_location} | Ped@{ped_loc}")

                self.ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

                # Compose + render (single window)
                composed = self._compose_tiled_frame()
                if composed is not None:
                    self.ego_viz.rgb_array = composed
                self.ego_viz.render(throttle, steer, brake, sim_time)

                # Exit criteria
                if self.first_brake_time is not None:
                    timed_out = (sim_time - self.first_brake_time) > post_brake_timeout_s
                    if self.stopped_time is not None or timed_out:
                        break

                time.sleep(0.005)

        except KeyboardInterrupt:
            print("[Client] Interrupted.")
        finally:
            self.cleanup()

    # ---------------------------
    # Logging
    # ---------------------------
    def _log_image_row(self, *, sim_time, device_id, frame_id,
                   capture_time, capture_loc, send_loc, speed_mps):
        # ego pose now (log-time)
        ego_loc_now = self.ego.get_location()
        ego_speed_mps_log = float(np.linalg.norm([
            self.ego.get_velocity().x, self.ego.get_velocity().y, self.ego.get_velocity().z
        ]))
        ego_speed_mps_capture = speed_mps if speed_mps is not None else ego_speed_mps_log

        # GT ped state (evaluation only)
        p = getattr(self, 'gt_ped_loc', None)
        v = getattr(self, 'gt_ped_vel', None)
        p_x, p_y, p_z = (p if p is not None else (None, None, None))
        v_px, v_py, v_pz = (v if v is not None else (None, None, None))
        ped_speed_mps = None if v is None else float((v_px**2 + v_py**2 + v_pz**2) ** 0.5)

        # start time/flag consistent with walker loop
        ped_start_time = float(self._walker_t_start + 4.0)
        ped_started = (sim_time >= ped_start_time) or (ped_speed_mps or 0.0) > 0.1

        self.logger.writerow({
            "row_type": "image",
            "sim_time": round(sim_time, 6),
            "device_id": device_id,
            "frame_id": int(frame_id),
            "frame_capture_time": round(capture_time, 6),
            "frame_capture_loc": capture_loc,
            "frame_send_time": round(time.time() - self.start_time, 6),
            "frame_send_loc": send_loc,
            "server_processing_time": None,
            "server_response_receive_time": None,
            "server_response_loc": None,
            "brake_issued": bool(self.first_brake_time is not None),

            # ego pose (log-time) — fill for BOTH ego/infra rows for consistency
            "x_ego": round(ego_loc_now.x, 2),
            "y_ego": round(ego_loc_now.y, 2),
            "z_ego": round(ego_loc_now.z, 2),

            # two speeds to avoid ambiguity
            "ego_speed_mps_capture": round(ego_speed_mps_capture, 3),
            "ego_speed_mph_capture": round(ego_speed_mps_capture * 2.23694, 2),
            "ego_speed_mps_log": round(ego_speed_mps_log, 3),
            "ego_speed_mph_log": round(ego_speed_mps_log * 2.23694, 2),

            # GT ped
            "p_x": p_x, "p_y": p_y, "p_z": p_z,
            "v_px": v_px, "v_py": v_py, "v_pz": v_pz,
            "ped_speed_mps": None if ped_speed_mps is None else round(ped_speed_mps, 3),
            "ped_started": ped_started,
            "ped_start_time": round(ped_start_time, 3),

            # crossing (echo params)
            "crossing_x": float(getattr(params, "crossing_x", -123.0)),
            "crossing_y": float(getattr(params, "crossing_y", -65.0)),
            "crossing_z": float(getattr(params, "crossing_z", 0.0)),
        })
        self.log_file.flush()


    def _log_cmd(self, sim_time, cmd, ego_loc_tuple):
        # current ego pose & log-time speed
        vel = self.ego.get_velocity()
        spd_log = float((vel.x**2 + vel.y**2 + vel.z**2) ** 0.5)

        row = {
            "row_type": "cmd",
            "sim_time": round(sim_time, 6),
            "device_id": "ego",
            "frame_id": int(cmd.get("frame_id", -1)),
            "frame_capture_time": None,
            "frame_capture_loc": None,
            "frame_send_time": None,
            "frame_send_loc": None,
            "server_processing_time": float(cmd.get("server_processing_time", 0.0)),
            "server_response_receive_time": round(time.time() - self.start_time, 6),
            "server_response_loc": ego_loc_tuple,
            "brake_issued": True,

            # ego pose (log-time)
            "x_ego": ego_loc_tuple[0], "y_ego": ego_loc_tuple[1], "z_ego": ego_loc_tuple[2],

            # use ONLY the new speed columns (no 'speed_mps'/'speed_mph' keys)
            "ego_speed_mps_capture": None,
            "ego_speed_mph_capture": None,
            "ego_speed_mps_log": round(spd_log, 3),
            "ego_speed_mph_log": round(spd_log * 2.23694, 2),

            # ped + crossing (safe defaults for cmd row)
            "p_x": None, "p_y": None, "p_z": None,
            "v_px": None, "v_py": None, "v_pz": None,
            "ped_speed_mps": None, "ped_started": None, "ped_start_time": None,
            "crossing_x": float(getattr(params, "crossing_x", -123.0)),
            "crossing_y": float(getattr(params, "crossing_y", -65.0)),
            "crossing_z": float(getattr(params, "crossing_z", 0.0)),
        }
        self.logger.writerow(row)
        self.log_file.flush()


    # ---------------------------
    # Cleanup
    # ---------------------------
    def cleanup(self):
        try:
            if self.ego_cam: self.ego_cam.destroy()
            if self.infra_cam: self.infra_cam.destroy()
        except Exception:
            pass
        try:
            self.ego_sock.close()
            if self.infra_sock: self.infra_sock.close()
        except Exception:
            pass
        # Stop walker thread
        try:
            self._walker_run = False
            if self._walker_thread and self._walker_thread.is_alive():
                self._walker_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.ego_viz.close()
        except Exception:
            pass
        try:
            self.log_file.close()
            print(f"[Client] Log saved: {LOG_PATH}")
        except Exception:
            pass


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    app = V2IClient(world)
    app.run()

if __name__ == "__main__":
    main()
