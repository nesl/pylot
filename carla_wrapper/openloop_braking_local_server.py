import socket
import pickle
import zlib
import time
import os
import cv2
import random
from ultralytics import YOLO
import numpy as np
from utils.service import recv_msg, send_msg
import braking_params as params

# ==== Experiment Parameters (Same as Client) ====
VEHICLE_NAME = params.vehicle_name
VEHICLE_SPEED = params.vehicle_speed
OBJECT_TYPE = params.object_type
PLATFORM = params.platform
MODEL = params.model
# ================================================

DET_BASE_DIR = "final/obs_type"#"all_yolo_braking_results"
DET_SUBDIR = f"{MODEL}_{PLATFORM}_server_detections/{VEHICLE_NAME}_{VEHICLE_SPEED}_{OBJECT_TYPE}"
DET_FULL_PATH = os.path.join(DET_BASE_DIR, DET_SUBDIR)
os.makedirs(DET_FULL_PATH, exist_ok=True)

TARGET_CLASSES = ['car', 'motorcycle', 'bicycle', 'person']  # classes that trigger braking

def draw_detections(image, detections, class_names):
    for box in detections[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{class_names[cls]} {conf:.2f}"
        bbox = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image

def frame_receiver_server():
    use_diff = False  # Set to False to ignore XOR compression
    prev_frame = None
    consecutive_detections = 0
    DETECTION_THRESHOLD = 3
    model = YOLO(f"{MODEL}.pt")
    class_names = model.names

    # Lookup class IDs for all target classes
    target_class_ids = [k for k, v in class_names.items()
                        if any(target in v.lower() for target in TARGET_CLASSES)]

    if not target_class_ids:
        print("[Server] Error: None of the target classes were found in YOLO class names!")
        return

    host, port = params.server_host, params.server_port
    sock = socket.socket()
    sock.bind((host, port))
    sock.listen(1)

    print("[Server] Waiting for connection...")
    conn, addr = sock.accept()
    print(f"[Server] Connected by {addr}")

    while True:
        try:
            data = recv_msg(conn)
            if data is None:
                print("[Server] Client disconnected.")
                break

            t_start = time.time()
            message = pickle.loads(zlib.decompress(data))
            frame_id = message.get('frame_id', -1)
            timestamp = message['timestamp']
            frame_type = message.get('frame_type', 'I')  # default to 'I' if key missing
            jpeg_bytes = message['rgb_frame']
            # decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)

            if use_diff and frame_type == 'D':
                if prev_frame is None:
                    raise RuntimeError(f"[Server] Received D-frame {frame_id} but no previous frame to diff against.")
                rgb_frame = cv2.bitwise_xor(decoded, prev_frame)
                print(f"[Server] Reconstructed D-frame {frame_id} at t={timestamp:.2f}s")
            else:
                rgb_frame = jpeg_bytes #decoded
                print(f"[Server] Received I-frame {frame_id} at t={timestamp:.2f}s")

            # prev_frame = rgb_frame.copy()

            print(f"[Server] Received frame {frame_id} at t={timestamp:.2f}s")

            # Simulate small processing delay
            time.sleep(0.013)
            detections = model.predict(rgb_frame, verbose=False)

            detection_found = False

            # Check if any of the target classes are detected within ROI
            for det in detections[0].boxes:
                cls = int(det.cls[0])
                conf = float(det.conf[0])
                bbox = det.xyxy[0].cpu().numpy()
                cx = (bbox[0] + bbox[2]) / 2

                if cls in target_class_ids and 300 <= cx <= 500 and conf > 0.3:
                    # t_end = time.time()
                    # processing_time = t_end - t_start
                    label = class_names[cls]
                    detection_found = True
                    print(f"[Server] Detected {label} in frame {frame_id}")

            # Update consecutive detection counter
            if detection_found:
                consecutive_detections += 1
                print(f"[Server] Consecutive detections: {consecutive_detections}")
            else:
                consecutive_detections = 0

            # Issue brake only after N consistent detections
            if consecutive_detections >= DETECTION_THRESHOLD:
                processing_time = time.time() - t_start
                response = {
                    'brake': True,
                    'server_processing_time': processing_time,
                    'frame_id': frame_id
                }
                serialized = zlib.compress(pickle.dumps(response))
                send_msg(conn, serialized)
                print(f"[Server] Brake issued at frame {frame_id} after {DETECTION_THRESHOLD} consecutive detections")
                break

            # Save annotated frame
            annotated_img = draw_detections(rgb_frame.copy(), detections, class_names)
            annotated_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
            out_path = os.path.join(DET_FULL_PATH, f"frame_{frame_id:05d}.jpg")
            cv2.imwrite(out_path, annotated_bgr)

        except Exception as e:
            print("[Server] Error:", e)
            break

    conn.close()
    sock.close()
    print("[Server] Shutdown")

if __name__ == "__main__":
    frame_receiver_server()
