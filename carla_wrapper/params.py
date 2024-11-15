# Scenario params

tracker_type    = 'sort' # ['sort', 'deep_sort']
prediction_type = 'linear' # ['linear', 'r2p2']
planner_type    = 'waypoints' # ['waypoints', 'fot', 'hybrid', 'rrtstar']
controller_type = 'mpc' # ['pid', 'mpc']

distributed = True # [True, False]
perception_loc  = 'local' # ['local', 'cloud']
control_loc = 'local' # ['local', 'cloud']

local_server = '10.0.0.4'
local_port = 5010
cloud_server = '0.0.0.0'
cloud_port = 5002

deadline_enforcement = 'none' # ['none', 'static', 'dynamic']
tracking_deadline = None
planning_deadline = None

detector_type = 'yolo' 

# Simulator params

simulator_host = '127.0.0.1'
simulator_port = 2000
simulator_timeout = 10

simulator_camera_frequency = -1

camera_image_width = 960
camera_image_height = 512
camera_fov = 90.0

simulator_control_frequency = -1
simulator_fps = 10

# Service params

host="0.0.0.0"
port=5010

device='cpu' # used by torch for r2p2 computation

# Taken from control/flags.py

min_pid_steer_waypoint_distance = 5
min_pid_speed_waypoint_distance = 5

stop_for_traffic_lights = False
stop_for_people = False
stop_for_vehicles = False
stop_at_uncontrolled_junctions = False

traffic_light_min_distance = 5
traffic_light_max_distance = 20
traffic_light_max_angle = 0.6
vehicle_max_distance = 15
vehicle_max_angle = 0.4
person_distance_hit_zone = 35
person_angle_hit_zone = 0.15
person_distance_emergency_zone = 15
person_angle_emergency_zone = 0.5

throttle_max = 1.0
steer_gain = 0.5
brake_max = 1.0
coast_factor = 1.75

# Taken from prediction/flags.py

prediction_radius = 50
prediction_num_past_steps = 5
prediction_num_future_steps = 30
prediction_ego_agent = False
r2p2_model_path = '/home/erdos/workspace/pylot/dependencies/models/prediction/r2p2/r2p2-model.pt'

# Taken from planning/flags.py
target_speed = 10.0
obstacle_radius = 1.0
num_waypoints_ahead = 30
num_waypoints_behind = 10
obstacle_filtering_distance = 1.0

# Taken from perception/flags.py

static_obstacle_distance_threshold = 50.0
dynamic_obstacle_distance_threshold = 75.0

traffic_light_det_min_score_threshold = 0.3

tracking_num_steps = 10

min_matching_iou=0.5
obstacle_track_max_age=3
deep_sort_tracker_weights_path='/home/erdos/workspace/pylot/dependencies/models/tracking/deep-sort-carla/feature_extractor'

obstacle_detection_gpu_index = 0
obstacle_detection_model_paths="/home/erdos/workspace/pylot/dependencies/models/obstacle_detection/faster-rcnn"
path_coco_labels = "/home/erdos/workspace/pylot/dependencies/models/pylot.names"
obstacle_detection_min_score_threshold = 0.5

timestamp_log_file = '/home/erdos/workspace/pylot/carla_wrapper/timestamp_log.txt'
