# ===============================
# v2i_braking_params.py (Python)
# Vehicle-to-Infrastructure demo
# ===============================

# ---------- Platform → server host/port ----------
platform_server_map = {
    "local":  {"host": "127.0.0.1", "port": 11000},
    "edge":   {"host": "10.0.0.6",  "port": 11000},
    "cloud":  {"host": "localhost", "port": 11000},
}

# Activate exactly one platform (kept like your v2v params)
only_cloud_active = True
only_edge_active  = False
only_local_active = False

_flags = [only_cloud_active, only_edge_active, only_local_active]
if sum(_flags) != 1:
    # Keep the constraint simple: exactly one True
    raise ValueError("Set exactly one of only_cloud_active / only_edge_active / only_local_active to True.")

if only_cloud_active:
    active_platforms = ["cloud"]
elif only_edge_active:
    active_platforms = ["edge"]
else:
    active_platforms = ["local"]

# ---------- Experiment axes (simple 1×1 defaults; extend as you like) ----------
vehicles      = ["audi.tt"]       # ego vehicle (ScenarioRunner will spawn by blueprint "vehicle.<name>")
speeds        = ["20mph"]         # ego target cruise for your client logic (informational here)
object_types  = ["person"]        # predominant obstacle class
models        = ["yolo11x"]       # kept for consistency with your server paths

# ---------- ScenarioRunner knobs (EGO + OCCLUDER + PEDESTRIAN) ----------
scenario_defaults = {
    "map_name":           "Town05",

    # Ego spawn (ScenarioRunner will spawn ego and leave it unmanaged for the client)
    "ego_spawn_x":        -122.958664,
    "ego_spawn_y":         130.0,
    "ego_spawn_z":          0.6,
    "ego_spawn_yaw":        -90.0,   # deg

    # Occluder (a large truck/van parked to block ego's line-of-sight)
    "occluder_model":      "vehicle.carlamotors.carlacola",
    "occluder_x":         -105.958664,
    "occluder_y":          -70,
    "occluder_z":           0.6,
    "occluder_yaw":        -90.0,

    # Pedestrian: spawn at edge, hold until t_ped_start_s, then cross at ped_speed_mps along yaw
    "ped_start_x":        -110.0,
    "ped_start_y":         -65.0,
    "ped_start_z":          0.6,
    "ped_start_yaw":       180.0,   # crossing direction
    "ped_speed_mps":        2.0,   # ~1.2–1.5 m/s
    "t_ped_start_s":        15.3,   # sim-time to start crossing

    # Scenario lifetime
    "scenario_timeout_s":  600.0,
}

# ---------- Client (streaming & camera knobs) ----------
client_defaults = {
    # Ego camera (roof/bonnet) — client will attach and stream to server
    "ego_camera_fov_deg":   60.0,
    "ego_stream_fps":       15,
    "ego_max_queue":         2,
    "stop_sending_on_brake": True,

    # Infrastructure pole camera (spawned/managed by client)
    "infra_enabled":         True,
    "infra_image_size":     (1280, 720),
    "infra_fov_deg":         70.0,
    "infra_height_m":         15.8,      # pole height
    "infra_pitch_deg":      -39.0,      # slight pitch down
    # Pole base (world) — choose a corner near the crossing; adjust to your map spot
    "infra_base_x":         -141.0,
    "infra_base_y":          -76.0,
    "infra_base_z":           0.0,
    "infra_yaw_deg":         0.0,
}

# ---------- Server knobs (two-camera intake; ego drives decisions first) ----------
server_defaults = {
    "roi_cx_min": 150,
    "roi_cx_max": 700,
    "roi_cy_min": 50,
    "roi_cy_max": 700,
    "roi_cx_min_infra": 850,
    "roi_cx_max_infra": 1150,
    "roi_cy_min_infra": 100,
    "roi_cy_max_infra": 600,
    "conf_thresh":          0.40,
    "use_consecutive":      False,
    "consecutive_n":        3,
    "prediction_horizon_s": 7.0,     # consider meets up to ~7s out
    "min_ped_speed_pxps":   1.5,     # allow modest apparent motion to yield ETA
    "ttc_thresh_s":         3.0,     # choose 2.0–3.0 per your risk tolerance
    "collision_window_s":   2.0,     # keep 0.8–1.0; TTC will usually trigger first anyway

    # Device identifiers (payload field from client → used to route logic)
    "ego_device_id":        "ego",
    "infra_device_id":      "infra",

    # Mode switch scaffold (future use): "baseline" uses ego frames only; "v2i" will read infra too
    "mode":                 "baseline_local",  # expected: 'baseline_local' | 'baseline_cloud' | 'v2i_local' | 'v2i_external'
}

# ---------- Build experiment matrix (kept identical in spirit to your v2v) ----------
def _derive_speed_mps(speed_str: str) -> float:
    if speed_str.endswith("kph"):
        return float(speed_str[:-3]) / 3.6
    if speed_str.endswith("mph"):
        return float(speed_str[:-3]) * 0.44704
    raise ValueError("vehicle_speed must end with 'kph' or 'mph'")

experiment_matrix = []
for m in models:
    for s in speeds:
        for v in vehicles:
            for o in object_types:
                for p in active_platforms:
                    entry = {
                        "vehicle_name":  v,          # e.g., "audi.tt"
                        "vehicle_speed": s,          # e.g., "25mph"
                        "object_type":   o,          # e.g., "person"
                        "model":         m,          # e.g., "yolo11x"
                        "platform":      p,          # cloud|edge|local
                    }
                    # Scenario defaults
                    entry.update({k: val for k, val in scenario_defaults.items()})
                    # Client defaults (prefixed to keep the shape)
                    entry.update({f"client_{k}": val for k, val in client_defaults.items()})
                    # Server defaults (prefixed as well)
                    entry.update({f"server_{k}": val for k, val in server_defaults.items()})

                    # For convenience, include derived ego speed in m/s for client control
                    entry["ego_speed_mps"] = _derive_speed_mps(s)

                    experiment_matrix.append(entry)

# Active experiment selection
active_index = 0
active_rep   = 1
active_config = experiment_matrix[active_index]

# ---------- Unpack core fields (backward-compatible with your code) ----------
vehicle_name   = active_config["vehicle_name"]
vehicle_speed  = active_config["vehicle_speed"]
object_type    = active_config["object_type"]
platform       = active_config["platform"]
model          = active_config["model"]

# Server endpoint
server_host = platform_server_map[platform]["host"]
server_port = platform_server_map[platform]["port"]

# Scenario (for ScenarioRunner)
map_name           = active_config["map_name"]
ego_spawn_x        = active_config["ego_spawn_x"]
ego_spawn_y        = active_config["ego_spawn_y"]
ego_spawn_z        = active_config["ego_spawn_z"]
ego_spawn_yaw      = active_config["ego_spawn_yaw"]

occluder_model     = active_config["occluder_model"]
occluder_x         = active_config["occluder_x"]
occluder_y         = active_config["occluder_y"]
occluder_z         = active_config["occluder_z"]
occluder_yaw       = active_config["occluder_yaw"]

ped_start_x        = active_config["ped_start_x"]
ped_start_y        = active_config["ped_start_y"]
ped_start_z        = active_config["ped_start_z"]
ped_start_yaw      = active_config["ped_start_yaw"]
ped_speed_mps      = active_config["ped_speed_mps"]
t_ped_start_s      = active_config["t_ped_start_s"]

scenario_timeout_s = active_config["scenario_timeout_s"]

# Client unpack (ego cam + infra cam)
ego_camera_fov_deg   = active_config["client_ego_camera_fov_deg"]
ego_stream_fps       = active_config["client_ego_stream_fps"]
ego_max_queue        = active_config["client_ego_max_queue"]
stop_sending_on_brake= active_config["client_stop_sending_on_brake"]

infra_enabled        = active_config["client_infra_enabled"]
infra_image_size     = active_config["client_infra_image_size"]
infra_fov_deg        = active_config["client_infra_fov_deg"]
infra_height_m       = active_config["client_infra_height_m"]
infra_pitch_deg      = active_config["client_infra_pitch_deg"]
infra_base_x         = active_config["client_infra_base_x"]
infra_base_y         = active_config["client_infra_base_y"]
infra_base_z         = active_config["client_infra_base_z"]
infra_yaw_deg        = active_config["client_infra_yaw_deg"]

# Server unpack
roi_cx_min           = active_config["server_roi_cx_min"]
roi_cx_max           = active_config["server_roi_cx_max"]
roi_cy_min           = active_config["server_roi_cy_min"]
roi_cy_max           = active_config["server_roi_cy_max"]
roi_cx_min_infra     = active_config["server_roi_cx_min_infra"]
roi_cx_max_infra     = active_config["server_roi_cx_max_infra"]
roi_cy_min_infra     = active_config["server_roi_cy_min_infra"]
roi_cy_max_infra     = active_config["server_roi_cy_max_infra"]
conf_thresh          = active_config["server_conf_thresh"]
use_consecutive      = active_config["server_use_consecutive"]
prediction_horizon_s = active_config["server_prediction_horizon_s"]
min_ped_speed_pxps   = active_config["server_min_ped_speed_pxps"]
ttc_thresh_s         = active_config["server_ttc_thresh_s"]
collision_window_s   = active_config["server_collision_window_s"]
consecutive_n        = active_config["server_consecutive_n"]
mode                 = active_config["server_mode"]

ego_device_id        = active_config["server_ego_device_id"]
infra_device_id      = active_config["server_infra_device_id"]
mode                 = active_config["server_mode"]   # "baseline" | "v2i"

# ---------- Brake controller parameters (same shape as your v2v) ----------
brake_controller_configs = {
    "audi.tt": {
        "target_decel": 6.0,
        "max_decel":    7.0,
        "kp":           0.2,
        "max_ramp":     0.1,
    },
    "carlamotors.carlacola": {
        "target_decel": 3.0,
        "max_decel":    3.0,
        "kp":           0.45,
        "max_ramp":     0.15,
    },
    "kawasaki.ninja": {
        "target_decel": 6.0,
        "max_decel":    6.0,
        "kp":           0.2,
        "max_ramp":     0.05,
    },
}
brake_params = brake_controller_configs.get(vehicle_name, brake_controller_configs["audi.tt"])
target_decel = brake_params["target_decel"]
max_decel    = brake_params["max_decel"]
kp           = brake_params["kp"]
max_ramp     = brake_params["max_ramp"]


# occlusion = "none" # none, light, heavy

# === Visibility preset selector (default) ===
occlusion = "clear"   # "none" | "light" | "high"

# === Weather profiles (CARLA WeatherParameters knobs) ===
WEATHER_PROFILES = {
    # Clear / baseline (unchanged)
    "clear": {
        "cloudiness": 0.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wetness": 0.0,
        "sun_altitude_angle": 50.0,
    },
    "light": {  # dusk + drizzle
        "cloudiness": 70.0,
        "precipitation": 30.0,
        "precipitation_deposits": 40.0,
        "wetness": 50.0,
        "sun_altitude_angle": 6.0,
    },
    "high": {  # night + storm
        "cloudiness": 100.0,
        "precipitation": 90.0,
        "precipitation_deposits": 90.0,
        "wetness": 100.0,
        "sun_altitude_angle": -5.0,
    }
}
