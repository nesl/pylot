# -- Platform-specific server configs --
platform_server_map = {
    "local":  {"host": "10.0.0.4", "port": 10000},
    "edge":   {"host": "10.0.0.6", "port": 10000}, 
    "cloud":  {"host": "localhost", "port": 10000}, 
}

# -- Platform activation flags (for testing subsets) --
only_cloud_active = True
only_edge_active = False
only_local_active = False

# -- Validate flags: only one should be True at a time --
platform_flags = [only_cloud_active, only_edge_active, only_local_active]
if sum(platform_flags) > 1:
    raise ValueError("Only one of the platform activation flags should be set to True.")

# -- Active platforms (based on flags) --
if only_cloud_active:
    active_platforms = ["cloud"]
elif only_edge_active:
    active_platforms = ["edge"]
elif only_local_active:
    active_platforms = ["local"]
else:
    active_platforms = ["cloud", "edge", "local"]

# ["audi.tt", "carlamotors.carlacola", "kawasaki.ninja"]
# -- All valid experiment combinations (vehicle × speed × object) --
vehicles = ["audi.tt", "carlamotors.carlacola", "kawasaki.ninja"]
speeds = ["40mph"]
object_types = ["car", "person", "bike"]
models = ["yolo11x"]

# -- Generate full experiment matrix based on active platforms --
experiment_matrix = [
    {"vehicle_name": v, "vehicle_speed": s, "object_type": o, "model": m, "platform": p}
    for m in models for s in speeds for v in vehicles for o in object_types
    for p in active_platforms
]

# -- Index of active experiment --
active_index = 0
active_rep = 1

# -- Active experiment configuration --
active_config = experiment_matrix[active_index]

# -- Unpack active values --
vehicle_name = active_config["vehicle_name"]
vehicle_speed = active_config["vehicle_speed"]
object_type = active_config["object_type"]
platform = active_config["platform"]
model = active_config["model"]

# -- Assign host and port based on platform --
server_host = platform_server_map[platform]["host"]
server_port = platform_server_map[platform]["port"]

# -- Brake controller parameters by vehicle --
brake_controller_configs = {
    "audi.tt": {
        "target_decel": 7.0,
        "max_decel": 7.3,
        "kp": 0.2,
        "max_ramp": 0.05
    },
    "carlamotors.carlacola": {
        "target_decel": 3.0,
        "max_decel": 3.0,
        "kp": 0.45,
        "max_ramp": 0.15
    },
    "kawasaki.ninja": {
        "target_decel": 6.0,
        "max_decel": 6.0,
        "kp": 0.2,
        "max_ramp": 0.05
    }
}

# -- Final unpacked control params --
brake_params = brake_controller_configs[vehicle_name]
target_decel = brake_params["target_decel"]
max_decel = brake_params["max_decel"]
kp = brake_params["kp"]
max_ramp = brake_params["max_ramp"]
