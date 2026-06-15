# v2v_braking_server.py  (distance + TTC gated)
import socket
import pickle
import zlib
import time
import os
import cv2
import numpy as np
from ultralytics import YOLO

from utils.service import recv_msg, send_msg
import v2v_braking_params as params

# ==== Experiment label pieces (same as client, for folder names) ====
VEHICLE_NAME  = params.vehicle_name
VEHICLE_SPEED = params.vehicle_speed
OBJECT_TYPE   = params.object_type
PLATFORM      = params.platform
MODEL         = params.model
MODE          = params.braking_mode
V2V_DELAY_S   = params.v2v_delay_s
NW_DELAY_S    = params.nw_delay_s

# ==== Output directory for annotated frames ====
DET_BASE_DIR  = "v2v"
if MODE == "v2v":
    DELAY = V2V_DELAY_S
elif MODE == "perception" and PLATFORM == "cloud":
    DELAY = NW_DELAY_S
else:
    DELAY = 0
DET_SUBDIR    = f"{MODE}_{MODEL}_{PLATFORM}_server_detections/{VEHICLE_NAME}_{VEHICLE_SPEED}_{DELAY}"
DET_FULL_PATH = os.path.join(DET_BASE_DIR, DET_SUBDIR)
os.makedirs(DET_FULL_PATH, exist_ok=True)

# ==== Target detector classes (kept broad; nearest match is used) ====
TARGET_CLASSES = ['car', 'motorcycle', 'bicycle', 'person']

# ==== Tunables from params with safe fallbacks ====
ROI_CX_MIN   = getattr(params, "roi_cx_min", 300)
ROI_CX_MAX   = getattr(params, "roi_cx_max", 500)
CONF_THRESH  = getattr(params, "conf_thresh", 0.30)

# Presence-based guard (optional; usually False for TTC mode)
USE_CONSEC   = getattr(params, "use_consecutive", False)
CONSEC_N     = getattr(params, "consecutive_n", 3) if USE_CONSEC else 1

# TTC gating knobs
TTC_THRESH_S = getattr(params, "ttc_thresh_s", 3.0)
K_TTC_FRAMES = getattr(params, "k_ttc_frames", 2)     # require TTC below threshold for K frames
ALPHA_TTC    = getattr(params, "alpha_ttc", 0.30)     # EMA smoothing

# Distance gating knobs
CAR_WIDTH_M  = getattr(params, "car_width_m", 1.85)   # Tesla Model 3-ish; override in params if needed
D_THRESH_M   = getattr(params, "d_thresh_m", 24.0)    # brake if < this and closing

# FOV fallback if client doesn't send it
FOV_DEFAULT_DEG = getattr(params, "camera_fov_deg", 60.0)

# =============================================================================
# Utilities
# =============================================================================
def focal_length_pixels(img_width_px: int, fov_deg: float) -> float:
    """Pinhole focal length in pixels from FOV and image width."""
    fov_rad = np.deg2rad(fov_deg)
    return (img_width_px / 2.0) / np.tan(fov_rad / 2.0)

def estimate_range_m(w_pixels: float, f_pixels: float, real_width_m: float) -> float:
    """Monocular range from apparent width."""
    w_pixels = max(1.0, float(w_pixels))
    return (f_pixels * real_width_m) / w_pixels

def draw_detections(image_rgb, detections, class_names, best_info=None, roi=None):
    """Draw all detections and overlay best_Z/TTC if provided."""
    img = image_rgb.copy()
    # ROI guide
    # if roi is not None:
    #     x1, x2 = roi
    #     cv2.line(img, (int(x1), 0), (int(x1), img.shape[0]-1), (255, 255, 0), 1)
    #     cv2.line(img, (int(x2), 0), (int(x2), img.shape[0]-1), (255, 255, 0), 1)

    for box in detections[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        label = f"{class_names[cls]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # if best_info is not None:
    #     Z, ema_ttc, v_rel, frame_id = best_info
    #     overlay = f"Z≈{Z:.1f}m  TTC≈{(ema_ttc if ema_ttc is not None else 999):.2f}s  v_rel≈{v_rel:.2f}m/s  fid={frame_id}"
    #     cv2.putText(img, overlay, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    return img

# =============================================================================
# Main server
# =============================================================================
def frame_receiver_server():
    # Load detector
    model = YOLO(f"{MODEL}.pt")
    class_names = model.names

    # Map target class names to YOLO class IDs
    target_class_ids = [k for k, v in class_names.items()
                        if any(t in v.lower() for t in TARGET_CLASSES)]
    if not target_class_ids:
        print("[Server] Error: No target classes found in YOLO names!")
        return

    # Bind socket
    host, port = params.server_host, params.server_port
    sock = socket.socket()
    sock.bind((host, port))
    sock.listen(1)
    print(f"[Server] Waiting for connection on {host}:{port} ...")
    conn, addr = sock.accept()
    print(f"[Server] Connected by {addr}")
    conn_start_time = time.time()

    # State
    consecutive_presence = 0
    ttc_below_counter    = 0
    prev_Z, prev_t       = None, None
    ema_ttc              = None

    while True:
        try:
            data = recv_msg(conn)
            if data is None:
                print("[Server] Client disconnected.")
                break

            t_start = time.time()
            message = pickle.loads(zlib.decompress(data))

            frame_id   = message.get('frame_id', -1)
            timestamp  = message.get('timestamp', 0.0)
            frame_type = message.get('frame_type', 'I')
            rgb_frame  = message['rgb_frame']

            # Optional ego/camera meta from client
            ego_pose      = message.get('ego_pose', None)
            ego_speed_mps = message.get('ego_speed_mps', None)
            image_size    = message.get('image_size', (rgb_frame.shape[1], rgb_frame.shape[0]))
            camera_fov    = message.get('camera_fov', FOV_DEFAULT_DEG)

            W, H = image_size
            f_px = focal_length_pixels(W, camera_fov)

            print(f"[Server] Received {frame_type}-frame {frame_id} at t={timestamp:.2f}s")
            time.sleep(NW_DELAY_S)

            # Run detector
            detections = model.predict(rgb_frame, verbose=False)

            # Find nearest valid detection inside ROI by estimated range
            best_Z = None
            best_bbox = None
            best_cls_name = None
            detection_found = False

            for det in detections[0].boxes:
                cls  = int(det.cls[0])
                conf = float(det.conf[0])
                x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
                cx = 0.5 * (x1 + x2)
                w_px = (x2 - x1)

                if cls in target_class_ids and (ROI_CX_MIN <= cx <= ROI_CX_MAX) and (conf > CONF_THRESH):
                    # Estimate range from apparent width
                    Z_est = estimate_range_m(w_px, f_px, CAR_WIDTH_M)
                    if best_Z is None or Z_est < best_Z:
                        best_Z = Z_est
                        best_bbox = (x1, y1, x2, y2)
                        best_cls_name = class_names[cls]
                    detection_found = True

            # Presence counter (optional)
            if detection_found:
                consecutive_presence += 1
            else:
                consecutive_presence = 0

            # Default: don't brake yet
            should_brake = False
            v_rel = 0.0

            # Distance/TTC logic only if we have a valid lead estimate
            now_t = time.time()
            if detection_found and best_Z is not None:
                if prev_Z is not None and prev_t is not None:
                    dt = max(1e-3, now_t - prev_t)
                    v_rel = max(0.0, (prev_Z - best_Z) / dt)  # closing speed only (>=0)

                    # Distance gate: within threshold and closing
                    sim_time = now_t-conn_start_time
                    print("sim_time: ",sim_time)
                    print("best_Z: ", best_Z)
                    print("D_THRESH_M: ", D_THRESH_M)
                    print("v_rel: ", v_rel)
                    if (best_Z < D_THRESH_M) and (v_rel > 0.0):
                        # if frame_id > 100:
                        if sim_time > 12:
                            print("Braking because of z")
                            should_brake = True

                    # TTC gate: EMA + K-frame confirmation
                    if v_rel > 1e-3:
                        ttc = best_Z / v_rel
                        ema_ttc = ttc if ema_ttc is None else (ALPHA_TTC * ttc + (1 - ALPHA_TTC) * ema_ttc)
                        print("ttc: ", ema_ttc)
                        if ema_ttc < TTC_THRESH_S:
                            ttc_below_counter += 1
                        else:
                            ttc_below_counter = 0

                        if ttc_below_counter >= max(1, int(K_TTC_FRAMES)):
                            should_brake = True
                            print("Braking because of ttc")
                    else:
                        # Not closing; reset TTC counter
                        ttc_below_counter = 0

                prev_Z, prev_t = best_Z, now_t

            # (Optional) legacy presence-only guard, if user wants it on top
            if USE_CONSEC and (consecutive_presence >= CONSEC_N):
                should_brake = True

            # Save annotated frame (with Z/TTC overlays if we had a best detection)
            annotated = draw_detections(
                rgb_frame,
                detections,
                class_names,
                best_info=(best_Z, ema_ttc, v_rel, frame_id) if best_Z is not None else None,
                roi=(ROI_CX_MIN, ROI_CX_MAX),
            )
            cv2.imwrite(os.path.join(DET_FULL_PATH, f"frame_{frame_id:05d}.jpg"),
                        cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            # Decide + respond
            if should_brake:
                processing_time = time.time() - t_start
                response = {
                    'brake': True,
                    'server_processing_time': processing_time,
                    'frame_id': frame_id
                }
                send_msg(conn, zlib.compress(pickle.dumps(response)))
                print(f"[Server] BRAKE issued at frame {frame_id} | Z≈{(best_Z or -1):.1f} m | "
                    f"TTC≈{(ema_ttc if ema_ttc is not None else 999):.2f}s | v_rel≈{v_rel:.2f} m/s")

                # ---- Add one of these before break ----
                # Option A: tiny grace sleep
                time.sleep(0.2)

                # Option B (better): wait for a 1-byte ACK from client, with timeout
                conn.settimeout(0.5)
                try:
                    _ = recv_msg(conn)  # expect small "ACK" if you implement it on client
                except Exception:
                    pass

                break

        except Exception as e:
            print("[Server] Error:", e)
            break

    conn.close()
    sock.close()
    print("[Server] Shutdown")


if __name__ == "__main__":
    frame_receiver_server()
