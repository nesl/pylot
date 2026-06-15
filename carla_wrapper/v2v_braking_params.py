# ============================
# braking_params.py  (UNIFIED)
# ============================

# -- Platform-specific server configs --
platform_server_map = {
    "local":  {"host": "10.0.0.4", "port": 10000},
    "edge":   {"host": "10.0.0.6", "port": 10000},
    "cloud":  {"host": "localhost", "port": 10000},
}

# -- Platform activation flags (only one True is allowed for quick testing) --
only_cloud_active = True
only_edge_active  = False
only_local_active = False

# -- Validate flags: only one should be True at a time --
_platform_flags = [only_cloud_active, only_edge_active, only_local_active]
if sum(_platform_flags) > 1:
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

# ---------------------------------
# Experiment axes (your originals)
# ---------------------------------
vehicles      = ["audi.tt"] #, "carlamotors.carlacola", "kawasaki.ninja"]
speeds        = ["20mph"]  # use "xxmph" or "yykph"
object_types  = ["car"] #, "person", "bike"]
models        = ["yolo11x"]

# ---------------------------------
# ScenarioRunner knobs (V1 scripted)
# ---------------------------------
scenario_defaults = {
    # V1 (leader) controls (used by ScenarioRunner)
    "speed_mps":        None,     # derived from vehicle_speed if None
    "headway_m":        30.0,
    "t_brake_sched_s":  6.0,
    "decel_mps2":      -5.5,
    "hold_v2_cruise":   True,     # KeepVelocity(V2) until your external controller takes over?

    # Optional V1 -> V2 event emit (you can ignore if not used yet)
    "emit_udp":         True,
    "emit_dest_ip":     "127.0.0.1",
    "emit_dest_port":   20001,
}

# ---------------------------------
# Client (V2) streaming knobs
# ---------------------------------
client_defaults = {
    "send_ego_meta":        True,   # include ego pose/speed/fov in each frame to help server
    "stop_sending_on_brake": True,  # stop TX once braking begins
    "stream_fps":           15,     # simple send rate governor (Hz)
    "max_queue":            2,      # keep only latest frames (drop stale)
    "camera_fov_deg":       60.0,   # must match sensor setting
}

# ---------------------------------
# Server perception knobs
# ---------------------------------
server_defaults = {
    # ROI for lead-car selection (center band). Keep as-is if you already tuned it.
    "roi_cx_min":      300,
    "roi_cx_max":      500,
    "conf_thresh":     0.30,

    # TTC-on-scale gate (recommended for follower scenario)
    "ttc_thresh_s":    2.0,   # brake when EMA(TTC) < this
    "k_ttc_frames":    2,     # frames below threshold before braking (set 1 for instant)
    "alpha_ttc":       0.30,  # EMA smoothing factor

    # Legacy presence guard (optional)
    "use_consecutive": False,
    "consecutive_n":   3,
}

# -------------------------------------------------------
# Build full experiment matrix (extended with new knobs)
# -------------------------------------------------------
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
                        "vehicle_name":  v,
                        "vehicle_speed": s,
                        "object_type":   o,
                        "model":         m,
                        "platform":      p,
                    }
                    # Attach copies of defaults so each entry is self-contained
                    entry.update({k: v for k, v in scenario_defaults.items()})
                    entry.update({f"client_{k}": v for k, v in client_defaults.items()})
                    entry.update({f"server_{k}": v for k, v in server_defaults.items()})
                    # Derive speed_mps if not explicitly set
                    if entry["speed_mps"] is None:
                        entry["speed_mps"] = _derive_speed_mps(s)
                    experiment_matrix.append(entry)

# -- Active experiment index & repetition --
active_index = 0
active_rep   = 1

# -- Active experiment configuration --
active_config = experiment_matrix[active_index]

# -- Unpack core fields (backward-compatible with your code) --
vehicle_name   = active_config["vehicle_name"]
vehicle_speed  = active_config["vehicle_speed"]
object_type    = active_config["object_type"]
platform       = active_config["platform"]
model          = active_config["model"]

# -- Assign host and port based on platform --
server_host = platform_server_map[platform]["host"]
server_port = platform_server_map[platform]["port"]

# -- ScenarioRunner fields (for your scripted leader V1) --
speed_mps        = active_config["speed_mps"]
headway_m        = active_config["headway_m"]
t_brake_sched_s  = active_config["t_brake_sched_s"]
decel_mps2       = active_config["decel_mps2"]
hold_v2_cruise   = active_config["hold_v2_cruise"]
emit_udp         = active_config["emit_udp"]
emit_dest_ip     = active_config["emit_dest_ip"]
emit_dest_port   = active_config["emit_dest_port"]
v2v_delay_s      = 0.02
nw_delay_s       = 0.02
braking_mode     = "v2v"

# -- Client streaming fields (prefixed in matrix; unpack here for convenience) --
send_ego_meta         = active_config["client_send_ego_meta"]
stop_sending_on_brake = active_config["client_stop_sending_on_brake"]
stream_fps            = active_config["client_stream_fps"]
max_queue             = active_config["client_max_queue"]
camera_fov_deg        = active_config["client_camera_fov_deg"]

# -- Server perception fields (prefixed in matrix; unpack here too) --
roi_cx_min      = active_config["server_roi_cx_min"]
roi_cx_max      = active_config["server_roi_cx_max"]
conf_thresh     = active_config["server_conf_thresh"]
ttc_thresh_s    = active_config["server_ttc_thresh_s"]
k_ttc_frames    = active_config["server_k_ttc_frames"]
alpha_ttc       = active_config["server_alpha_ttc"]
use_consecutive = active_config["server_use_consecutive"]
consecutive_n   = active_config["server_consecutive_n"]

# --------------------------------------------
# Brake controller parameters (unchanged)
# --------------------------------------------
brake_controller_configs = {
    "audi.tt": {
        "target_decel": 7.0,
        "max_decel":    7.3,
        "kp":           0.2,
        "max_ramp":     0.05,
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

# -- Final unpacked control params --
brake_params = brake_controller_configs[vehicle_name]
target_decel = brake_params["target_decel"]
max_decel    = brake_params["max_decel"]
kp           = brake_params["kp"]
max_ramp     = brake_params["max_ramp"]
