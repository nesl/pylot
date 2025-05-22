import params
import time
import numpy as np
import os
import logging
import utils.logging
import torch
#import tensorflow as tf

from objects.objects import BoundingBox2D, Obstacle, VEHICLE_LABELS
from objects.messages import ObstaclesMessage

from utils.detection_utils import OBSTACLE_LABELS
from utils.detection_utils import load_coco_labels, load_coco_bbox_colors

obstacle_detection_gpu_index = params.obstacle_detection_gpu_index
obstacle_detection_model_paths = params.obstacle_detection_model_paths
path_coco_labels = params.path_coco_labels
obstacle_detection_min_score_threshold = params.obstacle_detection_min_score_threshold

class ObjectDetector:

    def __init__(self):
        self._config_name = "ObjectDetector"
        self._module_logger = utils.logging.get_module_logger(self._config_name)
        self._csv_logger = utils.logging.ModuleCompletionLogger()

        print("\nInitializing Object Detector ... ")
        self._module_logger.info("\nInitializing Object Detector")
        
        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        #logging.getLogger('tensorflow').setLevel(logging.ERROR)
        
        # Only sets memory growth for flagged GPU
        #physical_devices = tf.config.experimental.list_physical_devices('GPU')
        #tf.config.experimental.set_visible_devices([physical_devices[obstacle_detection_gpu_index]], 'GPU')
        #tf.config.experimental.set_memory_growth(physical_devices[obstacle_detection_gpu_index], True)

        # Load the model from the saved_model format file.
        #self._tf_model = tf.saved_model.load(obstacle_detection_model_paths)

        self._coco_labels = load_coco_labels(path_coco_labels)
        self._bbox_colors = load_coco_bbox_colors(self._coco_labels)

        # Serve some junk image to load up the model.
        #self._run_tf_model(np.zeros((108, 192, 3), dtype='uint8'))

        if params.detector_type == 'yolo':
            torch.cuda.set_device(0)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(device)
            from ultralytics import YOLO
            self._model = YOLO('yolov8x.pt', task='detect')
            for i in range(10):
                self._model.predict('traffic-jam-getty.jpg', device=0)

        # Unique bounding box id. Incremented for each bounding box.
        self._unique_id = 0


    def get_obstacles(self, timestamp, frame):
        start_time = time.time()
        # The models expect BGR images.
        assert frame.encoding == 'BGR', 'Expects BGR frames'
        if params.detector_type == 'yolo':
            obstacles = self._run_yolo_model(frame.frame)
        
        #tf_obstacles = self._run_tf_model(frame.frame)
        #print(tf_obstacles)

        self._module_logger.info('@{}: {} obstacles: {}'.format(timestamp, self._config_name, obstacles))

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.log_module_completion(self._config_name, runtime)
        # Send out obstacles.
        return (timestamp, obstacles, runtime)

    def _run_yolo_model(self, frame):
        # Run batched inference on a list of images
        results = self._model.predict(
            source=frame[:,:,::-1],
            conf=0.1, #obstacle_detection_min_score_threshold,
            imgsz=[params.camera_image_height, params.camera_image_width],
            device=0,
            classes=[1,2,3,5,6,7])

        obstacles = []
        min_conf = 0.0
        # Process results list
        for result in results:
            names = result.names # map of all labels used for prediction e.g. {2: 'car'}
            boxes = result.boxes  # Boxes object for bounding box outputs
            label = ''
            index = 0
            if result.probs != None:
                index = probs.top1
                label = boxes.cls[index]
            else:
                for cls in boxes.cls:
                    if int(cls.item()) in names:
                        label = names[int(cls.item())]

            if (label in VEHICLE_LABELS):
                xyxy = boxes.xyxy[index]
                conf = boxes.conf[index].item()
                obstacles.append(
                    Obstacle(
                        bounding_box = BoundingBox2D(xyxy[0].item(), xyxy[2].item(), xyxy[1].item(), xyxy[3].item()),
                        confidence = conf,
                        label=label,
                        id=self._unique_id)
                        )
                self._unique_id += 1
        return obstacles

    def _run_tf_model(self, frame):
        # Expand dimensions since the model expects images to have
        # shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(frame, axis=0)

        infer = self._tf_model.signatures['serving_default']
        result = infer(tf.convert_to_tensor(value=image_np_expanded))

        boxes = result['boxes']
        scores = result['scores']
        classes = result['classes']
        num_detections = result['detections']

        num_detections = int(num_detections[0])
        res_classes = [int(cls) for cls in classes[0][:num_detections]]
        res_boxes = boxes[0][:num_detections]
        res_scores = scores[0][:num_detections]
        
        obstacles = []
        for i in range(0, num_detections):
            if res_classes[i] in self._coco_labels:
                if (res_scores[i] >= obstacle_detection_min_score_threshold):
                    if (self._coco_labels[res_classes[i]] in OBSTACLE_LABELS):
                        obstacles.append(
                            Obstacle(BoundingBox2D(
                                int(res_boxes[i][1] * params.camera_image_width),
                                int(res_boxes[i][3] * params.camera_image_width),
                                int(res_boxes[i][0] * params.camera_image_height),
                                int(res_boxes[i][2] * params.camera_image_height)),
                                     res_scores[i],
                                     self._coco_labels[res_classes[i]],
                                     id=self._unique_id)
                                )
                        self._unique_id += 1
            else:
                self._module_logger.warning('Filtering unknown class: {}'.format(res_classes[i]))
        
        return obstacles


# def main():
#     detector = ObjectDetector()
#     image = '/home/erdos/workspace/pylot/test_image.jpeg'
#     print(detector.get_obstacles(0, image))

# if __name__ == '__main__':
#     main()
