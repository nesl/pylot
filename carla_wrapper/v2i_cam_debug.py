#!/usr/bin/env python3
# Minimal infra-cam debug tool that mirrors your v2i_braking_params.py exactly.
import math, time, random
import numpy as np
import cv2
import carla

# ------------------ load your params (exact fields) ------------------
import v2i_braking_params as params

# Ego & occluder (exact names)
VEHICLE_NAME   = params.vehicle_name  # e.g., "audi.tt" (we'll prefix with "vehicle.")
EGO_SPAWN      = dict(x=params.ego_spawn_x, y=params.ego_spawn_y,
                      z=params.ego_spawn_z, yaw=params.ego_spawn_yaw)

OCCLUDER_BP    = params.occluder_model  # already full id, e.g., "vehicle.carlamotors.carlacola"
OCC_SPAWN      = dict(x=params.occluder_x, y=params.occluder_y,
                      z=params.occluder_z, yaw=params.occluder_yaw)

# Pedestrian start (no simulation; just spawn and leave static)
PED_START      = dict(x=params.ped_start_x, y=params.ped_start_y,
                      z=params.ped_start_z, yaw=params.ped_start_yaw)

# Infra camera (client_* unpacked in your params)
INFRA_SIZE      = tuple(params.infra_image_size)   # (W, H)
INFRA_FOV_DEG   = float(params.infra_fov_deg)
INFRA_HEIGHT_M  = float(params.infra_height_m)

# Live-editable camera state (based on your base+height semantics)
cam_state = {
    "x":     float(params.infra_base_x),
    "y":     float(params.infra_base_y),
    "z0":    float(params.infra_base_z),  # actual Z = z0 + INFRA_HEIGHT_M
    "pitch": float(params.infra_pitch_deg),
    "yaw":   float(params.infra_yaw_deg),
}

HELP = """
Controls:
  Move XY:     I / K  (Y+/Y-),   J / L  (X-/X+)
  Move Z:      U / O  (Z+/Z-)
  Rotate:      A / D (Yaw -/+),  W / S (Pitch +/ -)
  Toggle FOV frustum: F
  Snap spectator:  C (camera),  E (ego),  B (occluder),  R (ped)
  Print camera params: P  (paste back into v2i_braking_params.py)
  Quit: Q or ESC
"""
STEP_XY, STEP_Z, STEP_DEG = 0.25, 0.25, 2.0

# ------------------ small utils ------------------
def _mk_tf(d):
    return carla.Transform(
        carla.Location(float(d["x"]), float(d["y"]), float(d["z"])),
        carla.Rotation(yaw=float(d["yaw"]))
    )

def _cam_tf():
    return carla.Transform(
        carla.Location(x=cam_state["x"], y=cam_state["y"], z=cam_state["z0"] + INFRA_HEIGHT_M),
        carla.Rotation(pitch=cam_state["pitch"], yaw=cam_state["yaw"])
    )

def _cam_cb(image: carla.Image, latest):
    if len(image.raw_data) != image.height * image.width * 4: return
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
    latest["arr"] = arr

def _dist(a: carla.Location, b: carla.Location) -> float:
    dx, dy, dz = a.x - b.x, a.y - b.y, a.z - b.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def _annotate(frame, tf, ego=None, occ=None, ped=None):
    txt = f"x={tf.location.x:.1f} y={tf.location.y:.1f} z={tf.location.z:.1f}  pitch={tf.rotation.pitch:.1f} yaw={tf.rotation.yaw:.1f}"
    cv2.putText(frame, "INFRA " + txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    y0 = 58
    if ego is not None:
        cv2.putText(frame, f"dist->EGO: {_dist(tf.location, ego.get_location()):.1f} m",
                    (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA); y0 += 30
    if occ is not None:
        cv2.putText(frame, f"dist->OCCL: {_dist(tf.location, occ.get_location()):.1f} m",
                    (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA); y0 += 30
    if ped is not None:
        try:
            cv2.putText(frame, f"dist->PED: {_dist(tf.location, ped.get_location()):.1f} m",
                        (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,200), 2, cv2.LINE_AA)
        except Exception:
            pass
    return frame

def _draw_arrow(world, tf, life=0.25):
    dbg = world.debug
    loc = tf.location
    fwd = tf.rotation.get_forward_vector()
    head = carla.Location(x=loc.x + fwd.x*2.0, y=loc.y + fwd.y*2.0, z=loc.z + fwd.z*2.0)
    dbg.draw_arrow(loc, head, thickness=0.2, arrow_size=0.4, color=carla.Color(0,255,255), life_time=life)
    dbg.draw_string(loc + carla.Location(z=0.5),
                    f"INFRA pitch={tf.rotation.pitch:.1f} yaw={tf.rotation.yaw:.1f}",
                    color=carla.Color(255,255,0), life_time=life)

def _draw_frustum(world, tf, fov_deg, depth=30.0, life=0.25):
    dbg = world.debug
    loc, rot = tf.location, tf.rotation
    fwd, right, up = rot.get_forward_vector(), rot.get_right_vector(), rot.get_up_vector()
    half = math.tan(math.radians(fov_deg * 0.5))
    center = loc + carla.Location(x=fwd.x*depth, y=fwd.y*depth, z=fwd.z*depth)
    h = half * depth
    corners = [
        center + carla.Location(x=( right.x + up.x)*h, y=( right.y + up.y)*h, z=( right.z + up.z)*h),
        center + carla.Location(x=(-right.x + up.x)*h, y=(-right.y + up.y)*h, z=(-right.z + up.z)*h),
        center + carla.Location(x=( right.x - up.x)*h, y=( right.y - up.y)*h, z=( right.z - up.z)*h),
        center + carla.Location(x=(-right.x - up.x)*h, y=(-right.y - up.y)*h, z=(-right.z - up.z)*h),
    ]
    for c in corners:
        dbg.draw_line(loc, c, thickness=0.1, color=carla.Color(0,200,255), life_time=life)
    for i in range(4):
        dbg.draw_line(corners[i], corners[(i+1)%4], thickness=0.05, color=carla.Color(0,200,255), life_time=life)

def _print_cam_params(tf):
    z_base = tf.location.z - INFRA_HEIGHT_M
    print("\nPaste into v2i_braking_params.py:")
    print(f'infra_base_x = {tf.location.x:.2f}')
    print(f'infra_base_y = {tf.location.y:.2f}')
    print(f'infra_base_z = {z_base:.2f}')
    print(f'infra_height_m = {INFRA_HEIGHT_M:.2f}')
    print(f'infra_pitch_deg = {tf.rotation.pitch:.1f}')
    print(f'infra_yaw_deg = {tf.rotation.yaw:.1f}\n')

# ------------------ spawners (no behavior) ------------------
def _spawn_vehicle(world, blueprint_id: str, tf: carla.Transform):
    bp = world.get_blueprint_library().find(blueprint_id)
    return world.try_spawn_actor(bp, tf)

def _spawn_ped_static(world, start_tf: carla.Transform):
    """Spawn a walker (no controller) if available; else drop a traffic cone so the pose is visible."""
    bps = world.get_blueprint_library().filter("walker.pedestrian.*")
    if bps:
        bp = random.choice(bps)
        ped = world.try_spawn_actor(bp, start_tf)
        if ped: return ped
    # fallback marker
    try:
        cone_bp = world.get_blueprint_library().find("static.prop.trafficcone01")
        return world.try_spawn_actor(cone_bp, start_tf)
    except Exception:
        return None

# ------------------ main ------------------
def main():
    print(HELP)
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # ---- Camera ----
    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(INFRA_SIZE[0]))
    cam_bp.set_attribute("image_size_y", str(INFRA_SIZE[1]))
    cam_bp.set_attribute("fov", str(INFRA_FOV_DEG))
    cam = world.spawn_actor(cam_bp, _cam_tf())
    latest = {"arr": None}
    cam.listen(lambda image: _cam_cb(image, latest))

    # ---- Ego (exact blueprint from params.vehicle_name) ----
    # ego_bp_id = f"vehicle.{VEHICLE_NAME}"
    # ego = _spawn_vehicle(world, ego_bp_id, _mk_tf(EGO_SPAWN))
    # if ego is None:
    #     raise RuntimeError(f"[debug] could not spawn ego {ego_bp_id} at requested pose")

    # # ---- Occluder (exact blueprint already full in params) ----
    # occluder = _spawn_vehicle(world, OCCLUDER_BP, _mk_tf(OCC_SPAWN))
    # if occluder is None:
    #     raise RuntimeError(f"[debug] could not spawn occluder {OCCLUDER_BP} at requested pose")

    # # ---- Pedestrian (static at start pose) ----
    # ped = _spawn_ped_static(world, _mk_tf(PED_START))

    # ---- UI loop ----
    show_frustum, last_draw = True, 0.0
    try:
        while True:
            arr = latest["arr"]
            tf_now = cam.get_transform()

            # periodic world overlays
            now = time.time()
            if now - last_draw > 0.2:
                _draw_arrow(world, tf_now, life=0.25)
                if show_frustum: _draw_frustum(world, tf_now, INFRA_FOV_DEG, depth=30.0, life=0.25)
                # world.debug.draw_string(ego.get_location() + carla.Location(z=1.5), "EGO", life_time=0.25, color=carla.Color(255,255,0))
                # world.debug.draw_string(occluder.get_location() + carla.Location(z=1.5), "OCCLUDER", life_time=0.25, color=carla.Color(0,255,255))
                # if ped is not None:
                #     world.debug.draw_string(ped.get_location() + carla.Location(z=1.5), "PED", life_time=0.25, color=carla.Color(255,200,200))
                # last_draw = now

            if arr is not None:
                vis = _annotate(arr.copy(), tf_now, ego=ego, occ=occluder, ped=ped)
                cv2.imshow("V2I Infra Debug (params)", vis)
            else:
                blank = np.zeros((INFRA_SIZE[1], INFRA_SIZE[0], 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for frames...", (20, INFRA_SIZE[1]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,200), 2, cv2.LINE_AA)
                cv2.imshow("V2I Infra Debug (params)", blank)

            k = cv2.waitKey(1) & 0xFF
            moved = False
            if k in (27, ord('q'), ord('Q')): break

            if k in (ord('f'), ord('F')): show_frustum = not show_frustum
            if k in (ord('c'), ord('C')): world.get_spectator().set_transform(cam.get_transform())
            if k in (ord('e'), ord('E')): world.get_spectator().set_transform(ego.get_transform())
            if k in (ord('b'), ord('B')): world.get_spectator().set_transform(occluder.get_transform())
            if k in (ord('r'), ord('R')) and ped is not None: world.get_spectator().set_transform(ped.get_transform())
            if k in (ord('p'), ord('P')): _print_cam_params(cam.get_transform())

            # Move/rotate camera
            if k in (ord('j'), ord('J')): cam_state["x"] -= STEP_XY; moved = True
            if k in (ord('l'), ord('L')): cam_state["x"] += STEP_XY; moved = True
            if k in (ord('i'), ord('I')): cam_state["y"] += STEP_XY; moved = True
            if k in (ord('k'), ord('K')): cam_state["y"] -= STEP_XY; moved = True
            if k in (ord('u'), ord('U')): cam_state["z0"] += STEP_Z;  moved = True
            if k in (ord('o'), ord('O')): cam_state["z0"] -= STEP_Z;  moved = True
            if k in (ord('a'), ord('A')): cam_state["yaw"]   -= STEP_DEG; moved = True
            if k in (ord('d'), ord('D')): cam_state["yaw"]   += STEP_DEG; moved = True
            if k in (ord('w'), ord('W')): cam_state["pitch"] += STEP_DEG; moved = True
            if k in (ord('s'), ord('S')): cam_state["pitch"] -= STEP_DEG; moved = True

            if moved: cam.set_transform(_cam_tf())

    finally:
        try: cam.stop(); cam.destroy()
        except Exception: pass
        # for a in [ped, occluder, ego]:
        #     try:
        #         if a is not None and a.is_alive: a.destroy()
        #     except Exception:
        #         pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print(HELP)
    main()
