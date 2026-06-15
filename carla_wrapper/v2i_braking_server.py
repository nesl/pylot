# v2i_server.py
# RSU server: accepts TWO clients (ego + infra), runs detection on ego frames,
# and sends a BRAKE command back to the ego connection upon trigger.
#
# Networking/framing mirrors your V2V code:
#   - TCP sockets
#   - pickle.dumps(...) -> zlib.compress(...) -> send_msg/recv_msg

import socket
import pickle
import zlib
import time
import os
import cv2
import threading
import queue
import numpy as np
from collections import deque

from utils.service import recv_msg, send_msg
import v2i_braking_params as params  # ROI, consecutive-N, model, platform, etc.

# ---------- YOLO (optional). Falls back to a stub if model isn't available ----------
_MODEL = None
_MODEL_NAMES = None
try:
    from ultralytics import YOLO
    _MODEL = YOLO(f"{params.model}.pt")
    _MODEL_NAMES = _MODEL.names
except Exception as e:
    print("[RSU] YOLO not available; using stub detector:", e)
    _MODEL = None
    _MODEL_NAMES = {}

# ---------- Experiment/IO parameters ----------
VEHICLE_NAME = params.vehicle_name
VEHICLE_SPEED = params.vehicle_speed
OBJECT_TYPE  = params.object_type
PLATFORM     = params.platform
MODEL_NAME   = params.model
MODE         = params.mode  # expected: 'baseline_local' | 'baseline_cloud' | 'v2i_local' | 'v2i_external'

# Mode flags + ingress latency (ms)
IS_BASELINE  = str(MODE).lower().startswith("baseline")
IS_V2I       = str(MODE).lower().startswith("v2i")
EGO_NET_MS   = 50 if str(MODE).lower() == "baseline_cloud" else 0
INFRA_NET_MS = 50 if str(MODE).lower() == "v2i_external"  else 0

# Perception knobs (ROI & gating)
ROI_CX_MIN  = getattr(params, "roi_cx_min", 300)
ROI_CX_MAX  = getattr(params, "roi_cx_max", 500)
ROI_CY_MIN  = getattr(params, "roi_cy_min", None)
ROI_CY_MAX  = getattr(params, "roi_cy_max", None)

INFRA_ROI_CX_MIN  = getattr(params, "roi_cx_min_infra", 300)
INFRA_ROI_CX_MAX  = getattr(params, "roi_cx_max_infra", 500)
INFRA_ROI_CY_MIN  = getattr(params, "roi_cy_min_infra", None)
INFRA_ROI_CY_MAX  = getattr(params, "roi_cy_max_infra", None)

CONF_THRESH = getattr(params, "conf_thresh", 0.30)
USE_CONSEC  = getattr(params, "use_consecutive", False)
CONSEC_N    = getattr(params, "consecutive_n", 3) if USE_CONSEC else 1

INFRA_CROSSING_Y_PX = getattr(params, "infra_crossing_y_px", None)

# Collision decision knobs for prediction fusion
PREDICTION_HORIZON_S  = float(getattr(params, "prediction_horizon_s", 6.0))   # ignore ETAs beyond this
COLLISION_WINDOW_S    = float(getattr(params, "collision_window_s", 1.0))     # |egoETA - pedETA| < window
MIN_PED_SPEED_PXPS    = float(getattr(params, "min_ped_speed_pxps", 5.0))     # ignore stationary/noisy ped

# Brake ONLY for pedestrians
TARGET_CLASSES = ['person']

# ---- V2I TTC decision knobs (infra uses ego odometry) ----
CROSSING_X = float(getattr(params, "crossing_x", -124.96))  # set in params for your map
CROSSING_Y = float(getattr(params, "crossing_y", -65.0))
TTC_THRESH_S = float(getattr(params, "ttc_thresh_s", 3.0))  # brake if TTC < 3s
MIN_EGO_SPEED_MPS = float(getattr(params, "min_ego_speed_mps", 0.5))  # ignore if ego crawling

# Output directories
DET_BASE_DIR  = "v2i"
DET_SUBDIR    = f"{MODE}_{params.occlusion}_{MODEL_NAME}_{PLATFORM}_server_detections/{VEHICLE_NAME}_{VEHICLE_SPEED}"
DET_FULL_PATH = os.path.join(DET_BASE_DIR, DET_SUBDIR)
os.makedirs(DET_FULL_PATH, exist_ok=True)

# ---------- Utilities ----------
def _mph(mps: float) -> float:
    return float(mps) * 2.23693629

def _roi_for_device(device_id):
    """Return ROI bounds for the given device."""
    if device_id == "infra":
        return (INFRA_ROI_CX_MIN, INFRA_ROI_CX_MAX, INFRA_ROI_CY_MIN, INFRA_ROI_CY_MAX)
    # default: ego
    return (ROI_CX_MIN, ROI_CX_MAX, ROI_CY_MIN, ROI_CY_MAX)

def _center_in_roi_device(cx, cy, device_id):
    cx_min, cx_max, cy_min, cy_max = _roi_for_device(device_id)
    if cy_min is not None and cy_max is not None:
        return (cx_min <= cx <= cx_max) and (cy_min <= cy <= cy_max)
    return (cx_min <= cx <= cx_max)

def _center_in_roi(cx, cy):
    # Rectangular ROI (legacy helper for ego)
    if ROI_CY_MIN is not None and ROI_CY_MAX is not None:
        return (ROI_CX_MIN <= cx <= ROI_CX_MAX) and (ROI_CY_MIN <= cy <= ROI_CY_MAX)
    # Fallback to x-band (old behavior)
    return (ROI_CX_MIN <= cx <= ROI_CX_MAX)

def _target_class_ids():
    if not _MODEL_NAMES:
        return []
    ids = []
    for k, v in _MODEL_NAMES.items():
        name = v.lower()
        if any(t == name for t in TARGET_CLASSES):
            ids.append(k)
    return ids

_TARGET_IDS = _target_class_ids()

def _stub_detect(rgb_frame, device_id):
    h, w, _ = rgb_frame.shape
    cx_min, cx_max, _, _ = _roi_for_device(device_id)
    cx_band = rgb_frame[:, cx_min:cx_max]
    gray = cv2.cvtColor(cx_band, cv2.COLOR_RGB2GRAY)
    m = float(np.mean(gray))
    if m > 130.0:
        x1 = cx_min + int(0.25 * (cx_max - cx_min))
        x2 = cx_min + int(0.75 * (cx_max - cx_min))
        y1 = int(0.35 * h); y2 = int(0.85 * h)
        return [("person", 0.50, (x1, y1, x2, y2))]
    return []

def _yolo_detect(rgb_frame, device_id):
    det_out = _MODEL.predict(rgb_frame, verbose=False)
    boxes = det_out[0].boxes
    found = []
    for b in boxes:
        cls = int(b.cls[0]); conf = float(b.conf[0])
        if cls not in _TARGET_IDS or conf < CONF_THRESH:
            continue
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
        cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
        if _center_in_roi_device(cx, cy, device_id):
            found.append((_MODEL_NAMES[cls], conf, (x1, y1, x2, y2)))
    return found

def draw_detections(image, detections):
    img = image.copy()
    for label, conf, (x1, y1, x2, y2) in detections:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return img

class _CentroidTracker:
    """Single-target centroid tracker for 'person' already filtered by ROI."""
    def __init__(self, max_miss=5, hist_len=10):
        self.cx = None
        self.cy = None
        self.has_target = False
        self.misses = 0
        self.max_miss = max_miss
        self.hist = deque(maxlen=hist_len)  # entries: (t_seconds, cx, cy)

    def update(self, detections, t_now=None):
        if not detections:
            self.misses += 1
            if self.misses > self.max_miss:
                self.cx = self.cy = None
                self.has_target = False
                self.hist.clear()
            return False
        # pick highest-confidence box
        _, _, (x1, y1, x2, y2) = max(detections, key=lambda d: d[1])
        cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)
        if self.cx is None:
            self.cx, self.cy = cx, cy
        else:
            self.cx = 0.6 * self.cx + 0.4 * cx
            self.cy = 0.6 * self.cy + 0.4 * cy
        self.misses = 0
        self.has_target = True
        if t_now is not None:
            self.hist.append((float(t_now), float(self.cx), float(self.cy)))
        return True

# ---------- RSU Server ----------
class RSUServer:
    """
    Two-client TCP server. Each client sends dicts like:
      {'device_id': 'ego'|'infra', 'type':'frame', 'rgb_frame': ndarray, ...}

    We brake strictly on **ego** detections for PERSON inside ROI.
    """
    def __init__(self, host=None, port=None, save_frames=True, preview=False):
        self.host = host or params.server_host
        self.port = port or params.server_port
        self.save_frames = save_frames
        self.preview = preview

        # Sockets
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Connection registries
        self.conns = {}       # device_id -> socket
        self.threads = []     # client handler threads
        self.queues = {       # device_id -> Queue[(dict, t_recv)]
            "ego":   queue.Queue(maxsize=10),
            "infra": queue.Queue(maxsize=10)
        }

        # Detection/decision state
        self.consec = {"ego": 0, "infra": 0}
        self.brake_sent = False

        # Optional FPS governors for disk writes
        self.last_dump = {"ego": 0.0, "infra": 0.0}

        # Infra tracker (V2I decision path)
        self.tracker_infra = _CentroidTracker(max_miss=5, hist_len=12)

        # Ego telemetry cache (for TTC)
        self.ego_loc = None
        self.ego_speed = 0.0
        self.last_ego_time = 0.0

        # --- GT ped state (evaluation only; never used for decisions) ---
        self.gt_ped = None           # (px, py, pz)
        self.gt_ped_v = None         # (vx, vy, vz)
        self.gt_ped_time = None      # sim_time when GT sampled

        # --- async CSV logger for TTC/TTE/RM (evaluation only) ---
        self.log_q = queue.Queue(maxsize=1000)
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()

        self.frame_recv_time = None

        # --- async dump pipeline (moves annotation off the hot path) ---
        self.dump_q = queue.Queue(maxsize=100)
        self.dump_thread = threading.Thread(target=self._dump_worker, daemon=True)
        self.dump_thread.start()

        print(f"[RSU] Listening on {self.host}:{self.port}")

    # ----- Lifecycle -----
    def serve(self):
        self.sock.bind((self.host, self.port))
        self.sock.listen(2)

        acceptor = threading.Thread(target=self._accept_loop, daemon=True)
        acceptor.start()

        try:
            while True:
                # EGO: detect + decide (always)
                self._drain_and_process("ego", do_decide=True)
                # INFRA: only in V2I modes (fully ignored in baseline)
                if IS_V2I:
                    self._drain_and_process("infra", do_decide=True)
                time.sleep(0.002)
        except KeyboardInterrupt:
            print("[RSU] Ctrl-C received; shutting down.")
        finally:
            for c in list(self.conns.values()):
                try:
                    c.shutdown(socket.SHUT_RDWR)
                    c.close()
                except Exception:
                    pass
            try:
                self.sock.close()
            except Exception:
                pass
            print("[RSU] Shutdown complete.")

    def _accept_loop(self):
        while True:
            conn, addr = self.sock.accept()
            print(f"[RSU] New TCP client from {addr}")
            t = threading.Thread(target=self._client_handler, args=(conn,), daemon=True)
            t.start()
            self.threads.append(t)

    def _client_handler(self, conn: socket.socket):
        device_id = None
        try:
            while True:
                raw = recv_msg(conn)
                if raw is None:
                    print("[RSU] Client disconnected")
                    break
                msg = pickle.loads(zlib.decompress(raw))
                self.frame_recv_time = time.time()

                if not device_id:
                    device_id = msg.get("device_id", None)
                    if device_id not in ("ego", "infra"):
                        print("[RSU] Unknown device_id; closing.")
                        break
                    old = self.conns.get(device_id)
                    if old and old is not conn:
                        try:
                            old.shutdown(socket.SHUT_RDWR)
                            old.close()
                        except Exception:
                            pass
                    self.conns[device_id] = conn
                    print(f"[RSU] Registered connection for device_id='{device_id}'")

                # If this is the ego stream, cache odometry for TTC (accept old & new field names)
                if device_id == "ego":
                    ego_loc = msg.get("ego_loc", None)
                    if ego_loc is None and msg.get("ego_pose") is not None:
                        ep = msg["ego_pose"]
                        if isinstance(ep, (tuple, list)) and len(ep) >= 3:
                            ego_loc = (float(ep[0]), float(ep[1]), float(ep[2]))  # drop yaw if present

                    ego_spd = msg.get("ego_speed_mps", None)
                    sim_time = msg.get("sim_time", msg.get("timestamp", None))

                    if ego_loc is not None and ego_spd is not None and sim_time is not None:
                        self.ego_loc = tuple(ego_loc)
                        self.ego_speed = float(ego_spd)
                        self.last_ego_time = float(sim_time)
                    else:
                        print(f"[DBG][EGO][missing] keys={list(msg.keys())} "
                              f"ego_loc={ego_loc} ego_speed_mps={ego_spd} sim_time/timestamp={sim_time}")

                    # --- read GT ped pose/velocity for evaluation logging (never used in decisions) ---
                    gt_loc = msg.get("gt_ped_loc", None)
                    gt_vel = msg.get("gt_ped_vel_mps", None)
                    gt_t   = msg.get("sim_time", msg.get("timestamp", None))
                    if gt_loc is not None:
                        try:
                            self.gt_ped = (float(gt_loc[0]), float(gt_loc[1]), float(gt_loc[2]))
                        except Exception:
                            self.gt_ped = None
                    if gt_vel is not None:
                        try:
                            self.gt_ped_v = (float(gt_vel[0]), float(gt_vel[1]), float(gt_vel[2]))
                        except Exception:
                            self.gt_ped_v = None
                    if gt_t is not None:
                        try:
                            self.gt_ped_time = float(gt_t)
                        except Exception:
                            self.gt_ped_time = None

                # Debug print each time ego odometry is updated
                self._dbg_print_ego_state(when="odometry_update")

                # --- Baseline gating: ignore infra completely ---
                if device_id == "infra" and IS_BASELINE:
                    continue  # do not enqueue/process infra in baseline modes

                # --- Ingress latency injection (producer -> RSU) ---
                if device_id == "ego" and EGO_NET_MS > 0:
                    time.sleep(EGO_NET_MS / 1000.0)
                if device_id == "infra" and INFRA_NET_MS > 0:
                    time.sleep(INFRA_NET_MS / 1000.0)

                q = self.queues.get(device_id)
                if q:
                    # STRICT FIFO: never drop; apply backpressure to sender thread
                    q.put((msg, time.time()))
        except Exception as e:
            print("[RSU] Client handler error:", e)
        finally:
            if device_id and self.conns.get(device_id) is conn:
                self.conns.pop(device_id, None)
            try:
                conn.close()
            except Exception:
                pass

    # ----- Processing -----
    def _drain_and_process(self, device_id: str, do_decide: bool):
        q = self.queues[device_id]
        if q.empty():
            return

        # STRICT FIFO: process ALL queued frames this tick, oldest → newest
        while not q.empty():
            msg, t_recv = q.get()

            frame = msg.get("rgb_frame", None)
            frame_id = msg.get("frame_id", None)

            if not isinstance(frame, np.ndarray):
                continue
            
            frame_recv_time = time.time()

            # Single detection per frame (reuse for decision + dump)
            detections = self._detect(frame, device_id)

            # ---- Decision FIRST (hot path lean) ----
            if do_decide:
                if device_id == "ego":
                    detection_found = (len(detections) > 0)
                    self.consec[device_id] = self.consec[device_id] + 1 if detection_found else 0

                    # --- time-domain evaluation logging from EGO tick ---
                    TTC_e = self._ego_eta_s()
                    TTE_gt = self._tte_p_gt()
                    RM_gt = (TTC_e - TTE_gt) if (TTC_e is not None and TTE_gt is not None) else None
                    self._log_tick(sim_time=self.last_ego_time,
                                frame_id=frame_id,
                                source="ego",
                                TTC_e=TTC_e,
                                TTE_p_perc=None, RM_perc=None,   # no perceived ped in ego-only path
                                TTE_p_gt=TTE_gt, RM_gt=RM_gt,
                                reason="")

                    if detection_found and (self.ego_loc is not None) and (self.ego_loc[1] < -30):
                        label, conf, (x1, y1, x2, y2) = detections[0]
                        cx = 0.5 * (x1 + x2)
                        print(f"[RSU][{device_id}] Hit: {label} conf={conf:.2f} cx={cx:.1f} (frame={frame_id})")

                    if not self.brake_sent and self.consec[device_id] >= CONSEC_N and (self.ego_loc is not None) and (self.ego_loc[1] < -30):
                        t0 = time.time()
                        self._send_brake(frame_id=frame_id, reason="vision/ego", t_start=t0)
                        self.brake_sent = True

                elif device_id == "infra":
                    # Track & predict ped ETA (image space) at decision timebase
                    t_now = time.time()
                    self.tracker_infra.update(detections, t_now=t_now)

                    # Always compute ego ETA
                    ego_eta = self._ego_eta_s()

                    # Perceived ped ETA only if tracked
                    ped_eta = self._ped_eta_s_from_infra(frame.shape) if self.tracker_infra.has_target else None

                    # Optional prints
                    if self.tracker_infra.has_target:
                        print(f"[DBG][INFRA][frame={frame_id}] ped_centroid=(cx={self.tracker_infra.cx:.1f}, "
                            f"cy={self.tracker_infra.cy:.1f})")
                    else:
                        print(f"[DBG][INFRA][frame={frame_id}] no ped tracked in ROI")

                    print(f"[DBG][PRED] ego_eta_s={ego_eta:.2f}s" if ego_eta is not None else "[DBG][PRED] ego_eta_s=None")
                    print(f"[DBG][PRED] ped_eta_s={ped_eta:.2f}s" if ped_eta is not None else "[DBG][PRED] ped_eta_s=None")
                    if (ped_eta is not None) and (ego_eta is not None):
                        print(f"[DBG][FUSE] |ego-ped|={abs(ego_eta - ped_eta):.2f}s  window={COLLISION_WINDOW_S:.2f}s")

                    # Always run fusion with ped_eta possibly None (TTC-only guard will handle it)
                    self._predict_fuse_and_decide(ped_eta, ego_eta, frame_id)

                    # --- time-domain evaluation logging from INFRA tick ---
                    TTC_e = ego_eta
                    TTE_perc = ped_eta
                    RM_perc = (TTC_e - TTE_perc) if (TTC_e is not None and TTE_perc is not None) else None
                    TTE_gt = self._tte_p_gt()
                    RM_gt = (TTC_e - TTE_gt) if (TTC_e is not None and TTE_gt is not None) else None
                    self._log_tick(sim_time=self.last_ego_time,
                                frame_id=frame_id,
                                source="infra",
                                TTC_e=TTC_e,
                                TTE_p_perc=TTE_perc, RM_perc=RM_perc,
                                TTE_p_gt=TTE_gt, RM_gt=RM_gt,
                                reason="")


            # ---- Enqueue annotation AFTER decision (non-blocking, async writer) ----
            self._maybe_enqueue_dump(device_id, frame, frame_id, detections)

    def _detect(self, rgb_frame, device_id):
        """Device-aware detection dispatcher (YOLO or stub)."""
        if _MODEL is not None and _TARGET_IDS:
            return _yolo_detect(rgb_frame, device_id)
        return _stub_detect(rgb_frame, device_id)

    def _ttc_to_crossing(self, ego_loc_xyz, ego_speed_mps):
        if ego_speed_mps <= 1e-3:
            return None
        ex, ey, _ = ego_loc_xyz
        gap = ((CROSSING_X - ex)**2 + (CROSSING_Y - ey)**2)**0.5
        return gap / max(ego_speed_mps, 1e-3)

    def _ego_eta_s(self):
        """World-space ETA of ego to CROSSING (existing TTC)."""
        if (self.ego_loc is None) or (self.ego_speed < MIN_EGO_SPEED_MPS):
            return None
        return self._ttc_to_crossing(self.ego_loc, self.ego_speed)

    def _tte_p_gt(self):
        """
        Ground-truth TTE of the pedestrian to the crossing line (y = CROSSING_Y),
        using world pose/velocity sent by the client. Evaluation only.
        """
        if (self.gt_ped is None) or (self.gt_ped_v is None):
            return None
        py = float(self.gt_ped[1]); vy = float(self.gt_ped_v[1])
        dist = CROSSING_Y - py  # signed distance along y to the line
        # must be moving toward the line and have nonzero normal speed
        if vy == 0.0 or (dist * vy) <= 0.0:
            return None
        tte = abs(dist) / abs(vy)
        # keep same horizon as perceived ETA for apples-to-apples plots
        return tte if tte <= PREDICTION_HORIZON_S else None

    def _dbg_print_ego_state(self, when: str):
        if self.ego_loc is None:
            print(f"[DBG][EGO][{when}] no ego_loc yet")
            return
        ex, ey, ez = self.ego_loc
        eta = self._ego_eta_s()
        spd = self.ego_speed
        print(
            f"[DBG][EGO][{when}] loc=({ex:.2f},{ey:.2f},{ez:.2f}) "
            f"speed={spd:.2f}m/s ({_mph(spd):.1f}mph) ETA_to_crossing={eta:.2f}s"
            if eta is not None else
            f"[DBG][EGO][{when}] loc=({ex:.2f},{ey:.2f},{ez:.2f}) "
            f"speed={spd:.2f}m/s ({_mph(spd):.1f}mph) ETA_to_crossing=None"
        )

    def _ped_eta_s_from_infra(self, frame_shape):
        """
        Image-space ETA of ped (centroid y) to the configured crossing band in infra view.
        Constant-velocity model on smoothed centroid history.
        Returns seconds or None if not predictive.
        """
        if not self.tracker_infra.has_target or len(self.tracker_infra.hist) < 3:
            return None

        # Determine target Y (pixels) for crossing band
        h, w, _ = frame_shape
        if INFRA_CROSSING_Y_PX is not None:
            y_target = float(INFRA_CROSSING_Y_PX)
        else:
            # default: middle of infra ROI vertically (or image mid if no CY bounds)
            cy_min = INFRA_ROI_CY_MIN if INFRA_ROI_CY_MIN is not None else 0
            cy_max = INFRA_ROI_CY_MAX if INFRA_ROI_CY_MAX is not None else h
            y_target = 0.5 * (cy_min + cy_max)

        # Fit simple 1D velocity on y over last few points (robust to jitter)
        t_vals = [p[0] for p in self.tracker_infra.hist]
        y_vals = [p[2] for p in self.tracker_infra.hist]
        dt = t_vals[-1] - t_vals[0]
        if dt <= 1e-3:
            return None
        vy = (y_vals[-1] - y_vals[0]) / dt  # px/s

        # Must be moving toward the target and fast enough
        y_now = y_vals[-1]
        dist = (y_target - y_now)  # pixels; sign encodes direction
        moving_toward = (vy > 0 and dist > 0) or (vy < 0 and dist < 0)
        speed_pxps = abs(vy)
        if (not moving_toward) or (speed_pxps < MIN_PED_SPEED_PXPS):
            return None

        eta = abs(dist) / speed_pxps  # seconds
        return eta if eta <= PREDICTION_HORIZON_S else None

    def _predict_fuse_and_decide(self, ped_eta_s, ego_eta_s, frame_id):
        """
        Brake if (a) both ETAs exist and are within a collision window, OR
        (b) the imminent meet is within TTC_THRESH_S (2–3 s) by either ETA.
        """
        if self.brake_sent:
            return

        # If ego ETA is known and inside hard TTC threshold, brake even without ped ETA
        if (ego_eta_s is not None) and (ego_eta_s <= TTC_THRESH_S):
            self._send_brake(frame_id=frame_id, reason="v2i/predict_fusion", t_start=time.time())
            self.brake_sent = True
            return

        # From here on we need both ETAs for the |ΔETA| window / min() checks
        if (ped_eta_s is None) or (ego_eta_s is None):
            return

        if (ped_eta_s > PREDICTION_HORIZON_S) or (ego_eta_s > PREDICTION_HORIZON_S):
            return

        # (b) Hard TTC threshold — early “real-world” call (when both exist)
        if min(ped_eta_s, ego_eta_s) <= TTC_THRESH_S:
            self._send_brake(frame_id=frame_id, reason="v2i/predict_fusion", t_start=time.time())
            self.brake_sent = True
            return

        # (a) Existing “ETAs close” logic
        if abs(ped_eta_s - ego_eta_s) <= COLLISION_WINDOW_S:
            self._send_brake(frame_id=frame_id, reason="v2i/predict_fusion", t_start=time.time())
            self.brake_sent = True

    def _send_brake(self, frame_id, reason, t_start):
        ego_conn = self.conns.get("ego")
        if not ego_conn:
            print("[RSU] No ego conn available to send BRAKE.")
            return
        processing_time = time.time() - self.frame_recv_time
        response = {
            'brake': True,
            'server_processing_time': processing_time,
            'frame_id': frame_id,
            'reason': reason
        }
        # Extra debug context before issuing BRAKE
        if reason.startswith("v2i/"):
            ego_eta_dbg = self._ego_eta_s()
            print(
                f"[DBG][BRAKE_CTX] reason={reason} ego_eta={ego_eta_dbg:.2f}s"
                if ego_eta_dbg is not None else
                f"[DBG][BRAKE_CTX] reason={reason} ego_eta=None"
            )

        try:
            payload = zlib.compress(pickle.dumps(response))
            send_msg(ego_conn, payload)
            # --- log decision event (one-shot row) ---
            TTC_e_now = self._ego_eta_s()
            TTE_perc_now = None     # we don't recompute here; keep None
            RM_perc_now = None
            TTE_gt_now = self._tte_p_gt()
            RM_gt_now = (TTC_e_now - TTE_gt_now) if (TTC_e_now is not None and TTE_gt_now is not None) else None
            self._log_tick(sim_time=self.last_ego_time,
                        frame_id=frame_id,
                        source="decision",
                        TTC_e=TTC_e_now,
                        TTE_p_perc=TTE_perc_now, RM_perc=RM_perc_now,
                        TTE_p_gt=TTE_gt_now, RM_gt=RM_gt_now,
                        reason=reason)
            print(f"[RSU] >>> BRAKE issued (frame={frame_id}, reason={reason}, t_proc={processing_time:.3f}s)")
        except Exception as e:
            print("[RSU] Send BRAKE error:", e)

    # --- Async annotation dump helpers (10 Hz per device limiter preserved) ---
    def _maybe_enqueue_dump(self, device_id, rgb_frame, frame_id, dets):
        if not self.save_frames:
            return
        now = time.time()
        if now - self.last_dump[device_id] < 0.10:
            return
        self.last_dump[device_id] = now
        try:
            self.dump_q.put_nowait((device_id, int(frame_id), rgb_frame.copy(), dets or []))
        except queue.Full:
            # drop silently; never block the decision loop
            pass

    def _dump_worker(self):
        while True:
            device_id, frame_id, rgb_frame, dets = self.dump_q.get()
            try:
                ann = draw_detections(rgb_frame, dets)
                cx_min, cx_max, cy_min, cy_max = _roi_for_device(device_id)
                if cy_min is not None and cy_max is not None:
                    cv2.rectangle(ann, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 255), 2)
                else:
                    cv2.line(ann, (cx_min, 0), (cx_min, ann.shape[0]), (255, 255, 255), 2)
                    cv2.line(ann, (cx_max, 0), (cx_max, ann.shape[0]), (255, 255, 255), 2)
                if device_id == "infra":
                    h, w, _ = ann.shape
                    if INFRA_CROSSING_Y_PX is not None:
                        y_target = int(INFRA_CROSSING_Y_PX)
                    else:
                        cy_min = INFRA_ROI_CY_MIN if INFRA_ROI_CY_MIN is not None else 0
                        cy_max = INFRA_ROI_CY_MAX if INFRA_ROI_CY_MAX is not None else h
                        y_target = int(0.5 * (cy_min + cy_max))
                    cv2.line(ann, (0, y_target), (w, y_target), (0, 255, 255), 2)
                out_dir = os.path.join(DET_FULL_PATH, device_id)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"frame_{frame_id:05d}.jpg")
                cv2.imwrite(out_path, cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print("[RSU][dump_worker] error:", e)

    def _log_tick(self, *, sim_time, frame_id, source, TTC_e, TTE_p_perc, RM_perc, TTE_p_gt, RM_gt, reason=""):
        """
        Enqueue a non-blocking log row. `source` is 'ego' or 'infra' indicating who produced this tick.
        """
        try:
            self.log_q.put_nowait({
                "sim_time": sim_time,
                "frame_id": frame_id,
                "source": source,
                "mode": str(MODE),
                "ego_y": (self.ego_loc[1] if self.ego_loc else None),
                "ego_speed_mps": float(self.ego_speed),
                "TTC_e_s": (float(TTC_e) if TTC_e is not None else None),
                "TTE_p_perc_s": (float(TTE_p_perc) if TTE_p_perc is not None else None),
                "RM_perc_s": (float(RM_perc) if RM_perc is not None else None),
                "TTE_p_gt_s": (float(TTE_p_gt) if TTE_p_gt is not None else None),
                "RM_gt_s": (float(RM_gt) if RM_gt is not None else None),
                "reason": reason,
            })
        except queue.Full:
            pass  # drop log if saturated; never block

    def _log_worker(self):
        """
        Background writer: one CSV per run under DET_FULL_PATH, named 'risk_time_series.csv'.
        """
        import csv
        out_path = os.path.join(DET_FULL_PATH, "risk_time_series.csv")
        os.makedirs(DET_FULL_PATH, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "sim_time","frame_id","source","mode",
                "ego_y","ego_speed_mps",
                "TTC_e_s","TTE_p_perc_s","RM_perc_s","TTE_p_gt_s","RM_gt_s","reason"
            ])
            w.writeheader()
            while True:
                row = self.log_q.get()
                try:
                    w.writerow(row)
                    f.flush()
                except Exception as e:
                    print("[RSU][log_worker] error:", e)


# ---------- Entrypoint ----------
def main():
    host, port = params.server_host, params.server_port
    server = RSUServer(host=host, port=port, save_frames=True, preview=False)
    server.serve()

if __name__ == "__main__":
    main()
