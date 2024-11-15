import os
import cv2
import socket
import pickle
import time
import params
import numpy as np
from ultralytics import YOLO
from utils.service import send_msg, recv_msg
import zlib

actual_locations = {}
actual_speeds = {}
predicted_locations = {}
predicted_speeds = {}

overall_times = []
inference_times = []
input_processing_times = []
output_processing_times = []
transaction_times = []

metrics = {'fd_any': None, 'fd_car': None, 'fd_car_valid': None}

def get_actuator_instructions(speed):
    # Adjust based on speed or add logic if more instructions are needed
    return speed

# Define a function to save all detections in a specified directory
def save_detections_with_boxes(frame, detections, timestamp, output_dir="detections"):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use YOLOv8's plotting function to add bounding boxes
    plot_img = detections[0].plot()

    # Save the processed frame with a unique name based on timestamp
    #timestamp = int(time.time() * 1000)  # Use milliseconds timestamp for uniqueness
    output_path = os.path.join(output_dir, f"processed_frame_{timestamp}.jpg")
    cv2.imwrite(output_path, plot_img)
    print(f"Processed frame saved as: {output_path}")

def local_server(model):
    host = "0.0.0.0"
    port = 10000
    local_socket = socket.socket()
    local_socket.bind((host, port))

    print("Local server open...")

    local_socket.listen(10)
    local_conn, address = local_socket.accept()  

    positions = []
    timestamps = []

    first_detection = None  # Store the first detection details

    while True:
        # Receive data stream
        input_data = recv_msg(local_conn)
        #print("Sensor message received in:", time.time() - start_rx)

        overall_start_time = time.time()
        save_time = 0
        start_time = time.time()
        try:
            input_message = pickle.loads(zlib.decompress(input_data))
            timestamp = input_message['timestamp']
            rgb_frame = input_message['rgb_frame']
            depth_frame = input_message['depth_frame']
            camera_pose = input_message['pose']
            frame = input_message['frame']
        except (pickle.UnpicklingError, KeyError) as e:
            print("Error decoding input data:", e)
            continue
        
        input_processing_times.append(time.time() - start_time)

        # Run YOLOv8s model on the frame to detect objects
        start_time = time.time()
        detections = model.predict(rgb_frame, verbose=False)
        inference_times.append(time.time()-start_time)
        
        car_detected = False
        estimated_speed = 0
        bbox_center = None
        speed_set = False
        
        start_time = time.time()
        # calculate actual depth in meters
        # Calculate the normalized depth (0 to 1) from the RGB channels
        depth_in_meters = (depth_frame[:,:,0] + depth_frame[:,:,1] * 256.0 + depth_frame[:,:,2] * 256.0**2) / 16777.215 # CARLA scales depth by 1000

        for detection in detections[0].boxes:
            bbox = detection.xyxy[0].cpu().numpy()  # Get bounding box coordinates
            bbox_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            if bbox_center[0] > 500 or bbox_center[0] < 300:
                continue
            class_id = int(detection.cls[0])  # Get the class ID for each detection
            if metrics['fd_any'] == None:
                metrics['fd_any'] = {'frame': len(overall_times), 'class': class_id, 'conf': detection.conf}
            if class_id == 2 and metrics['fd_car'] == None:
                    metrics['fd_car'] = {'frame': len(overall_times), 'class': class_id, 'conf': detection.conf}
            if class_id == 2 and detection.conf >= 0.4: 
                if metrics['fd_car_valid'] == None:
                    metrics['fd_car_valid'] = {'frame': len(overall_times), 'class': class_id, 'conf': detection.conf} 
                car_detected = True
                #print(f"Car detected with bounding box: {bbox} and confidence: {detection.conf}")
                # Save the frame with bounding boxes and detections
                save_time = time.time()
                save_detections_with_boxes(frame, detections, timestamp, output_dir="detections")
                save_time = time.time() - save_time

                # Store the first detection details if not already stored
                if first_detection is None:
                    first_detection = {
                        "timestamp": timestamp,
                        "bbox": bbox,
                        "confidence": detection.conf
                    }
                    print(f"First car detection at timestamp: {first_detection['timestamp']} and frame {len(overall_times)}")

                break


        if car_detected and bbox_center:
            # Project 2D bounding box center to 3D world coordinates
            predicted_position = project_to_world(camera_pose, bbox_center, depth_in_meters)
            print(f"Car actual position    = {input_message['car_location']}")
            print(f"Car predicted position = {predicted_position} at frame {len(overall_times)}")
            predicted_locations[frame] = predicted_position
           
            if len(positions) != 0 and len(timestamps) != 0:
                if len(positions) > 2 and len(timestamps) > 2:
                    time_elapsed = timestamp - timestamps[0]
                    distance = np.linalg.norm(predicted_position - positions[0])
                    speed = distance / time_elapsed
                    speed_set = True
                    print(f"Car actual speed    = {input_message['car_velocity']}")
                    print(f"Car predicted speed = {speed} m/s over {distance}")
                    predicted_speeds[frame] = speed

            # Update last known position and timestamp for next calculation
            positions.append(predicted_position)
            timestamps.append(timestamp)

            # Output both the first and latest detection timestamps
            # print(f"First detection timestamp: {first_detection['timestamp']}, Latest detection timestamp: {timestamp}")
            # Send actuator instructions
            instruction = get_actuator_instructions(speed=estimated_speed)
            output_processing_times.append(time.time()-start_time-save_time)
            #print("Actuator instructions:", instruction)
            actuator_msg = instruction

            start_time = time.time()
            cmd = pickle.dumps(actuator_msg)        
            send_msg(local_conn, cmd)
            transaction_times.append(time.time() - start_time)
            #print("Actuator data sent in:", time.time() - start_tx)

        actual_locations[frame] = input_message['car_location']
        actual_speeds[frame] = np.linalg.norm(np.array(input_message['car_velocity']))

        overall_times.append(time.time() - overall_start_time - save_time)
        print(f"Average overall e2e time       = {np.mean(overall_times)}")
        print(f"Average input processing time  = {np.mean(input_processing_times)}")
        print(f"Average inference time         = {np.mean(inference_times)}")
        print(f"Average output processing time = {np.mean(output_processing_times)}")
        print(f"Average transaction time       = {np.mean(transaction_times)}")

        #else:
            #print("Car not detected in the frame.")

# Project 2D bbox center to 3D world coordinates
def project_to_world(camera_pose, bbox_center, camera_depth):
    depth_value = camera_depth[int(bbox_center[1]), int(bbox_center[0])]
    #print(f"bbox center: {bbox_center} and depth_value = {depth_value}")
    x, y = bbox_center
    normalized_coords = np.array([x, y, 1.0])
    
    width, height = 800, 600
    fov_rad = np.deg2rad(90) # assuming fov = 90
    # Calculate focal length in pixels
    focal_length = width / (2.0 * np.tan(fov_rad / 2.0))

    # Define the intrinsic matrix
    intrinsic_matrix = np.array([[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]])
    
    # Use intrinsic matrix to calculate 3D camera coordinates
    camera_point = np.linalg.inv(intrinsic_matrix) @ (normalized_coords * depth_value)
    #print(f"Location in pixel space = {normalized_coords}")
    #print(f"Location in camera space = {camera_point}")

    # Initialize a 4x4 identity matrix
    extrinsic_matrix = np.eye(4)
    
    # Set the rotation matrix (top-left 3x3 part)
    pitch, yaw, roll = camera_pose['rotation']
    pitch, yaw = np.deg2rad(-30), np.deg2rad(180)
    R_yaw = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    R_pitch = np.array([[1,0,0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    #R_roll = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    
    # Combined rotation matrix
    rotation_matrix = R_pitch @ R_yaw
    #rotation_matrix = [[-1,0,0], [0,0.86603,-0.5],[0,-0.5,-0.86603]] # R_pitch @ R_yaw
    #rotation_matrix = [[0,1,0], [0.86603,0,0.5],[-0.5,0,-0.86603]] # R_pitch @ R_yaw
    #rotation_matrix = [[0,-1,0], [0.5, 0, -0.86603],[0.86603, 0, 0.5]] # R_pitch @ R_yaw
    transform = np.array([[0,0,1], [1,0,0], [0,-1,0]])
    extrinsic_matrix[:3, :3] = rotation_matrix @ transform

    # Set the translation vector (top-right 3x1 part)
    position = np.array(camera_pose['position'])
    extrinsic_matrix[:3, 3] = position  # Use -R * position to align with world space

    #print(f"Camera location = {extrinsic_matrix[:3,3]}")

    # Convert the camera-space point to homogeneous coordinates
    camera_point_homogeneous = np.array([camera_point[0], camera_point[1], camera_point[2], 1.0])

    # Transform to world space by multiplying with the extrinsic matrix
    world_point_homogeneous = extrinsic_matrix @ camera_point_homogeneous

    # Drop the homogeneous component to return a 3D point in world space
    return world_point_homogeneous[:3] # @ np.array([[0,0,1], [1,0,0], [0,-1,0]])

    #transform = np.array([[0,0,1], [1,0,0], [0,-1,0]])
    #return camera_point @ transform + np.array(camera_pose['position'])

# Calculate speed in m/s between two 3D points
def calculate_speed(current_position, previous_position, time_elapsed):
    distance = np.linalg.norm(current_position - previous_position)
    return distance / time_elapsed

if __name__ == '__main__':

    # Initialize YOLOv8s model for object detection
    model = YOLO("yolo11x.pt")
    #model.export(format="engine")
    #model = YOLO("yolo11x.onnx", task='detect')
    
    for i in range(20):
        model.predict('./detections/processed_frame_18.05806255340576.jpg')
    try:
        local_server(model)
    except KeyboardInterrupt:
        metrics['overall_times'] = np.mean(overall_times)
        metrics['input_processing_times'] = np.mean(input_processing_times)
        metrics['inference_times'] = np.mean(inference_times)
        metrics['output_processing_times'] = np.mean(output_processing_times)
        metrics['transaction_times'] = np.mean(transaction_times)

        with open('data_dump.pkl', 'wb') as f:
            pickle.dump((metrics, actual_locations, predicted_locations, actual_speeds, predicted_speeds), f)

