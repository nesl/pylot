#!/usr/bin/env python3
# Export CARLA spawn visualization to files (PNG/SVG) + CSVs – no realtime rendering required.
import os
import time
import math
import csv
import numpy as np
import carla
import matplotlib.pyplot as plt

# ---------- user knobs ----------
OUTPUT_DIR = "./spawn_exports"
# If you only care about an area, set these; otherwise computed from driving waypoints.
REGION_OVERRIDE = None
# REGION_OVERRIDE = {"minx": -200, "maxx": 200, "miny": -200, "maxy": 200}
PED_STEP_M  = 2.5      # grid step for pedestrian sampling (lower = denser)
PED_PAD_M   = 20.0     # pad beyond driving bbox when sampling sidewalks
DRAW_TOPOLOGY = True   # draw a light road graph background

# Pull scenario map from params
try:
    import v2i_braking_params as params
    EXPECTED_MAP = getattr(params, "map_name", "Town05")  # e.g., "Town03"
except Exception:
    EXPECTED_MAP = "Town05"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def ensure_map(client, desired):
    client.set_timeout(10.0)
    world = client.get_world()
    if desired and desired not in world.get_map().name:
        world = client.load_world(desired)  # resets actors (fine; we only read the map)
    return world, world.get_map()

def driving_bbox(world_map):
    """Compute a bounding box from all driving waypoints to limit sidewalk sampling."""
    wps = world_map.generate_waypoints(10.0)  # ~10m spacing across all roads (Driving)
    xs = [wp.transform.location.x for wp in wps]
    ys = [wp.transform.location.y for wp in wps]
    return min(xs), max(xs), min(ys), max(ys)

def build_lane_mask():
    """Build a lane_type mask that works across CARLA versions."""
    lane_mask = carla.LaneType.Sidewalk
    if hasattr(carla.LaneType, "Crosswalk"):
        lane_mask |= carla.LaneType.Crosswalk
    else:
        if hasattr(carla.LaneType, "Shoulder"):
            lane_mask |= carla.LaneType.Shoulder
    if hasattr(carla.LaneType, "Parking"):
        lane_mask |= carla.LaneType.Parking
    return lane_mask

def export_vehicle_spawns(world_map, out_csv):
    spawns = world_map.get_spawn_points()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","x","y","z","yaw","pitch","roll"])
        for i, tf in enumerate(spawns):
            L = tf.location; R = tf.rotation
            w.writerow([i, round(L.x,3), round(L.y,3), round(L.z,3),
                        round(R.yaw,2), round(R.pitch,2), round(R.roll,2)])
    return spawns

def export_ped_samples(world_map, region, step_m, out_csv):
    minx, maxx, miny, maxy = region
    lane_mask = build_lane_mask()
    xs = np.arange(minx, maxx, step_m, dtype=float)
    ys = np.arange(miny, maxy, step_m, dtype=float)
    seen = set()
    pts = []
    for x in xs:
        for y in ys:
            loc = carla.Location(x=x, y=y, z=0.5)
            wp = world_map.get_waypoint(loc, project_to_road=True, lane_type=lane_mask)
            if not wp:
                continue
            L = wp.transform.location
            key = (round(L.x), round(L.y))  # coarse dedup for plotting/CSV size
            if key in seen:
                continue
            seen.add(key)
            pts.append((L.x, L.y))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x","y"])
        for x, y in pts:
            w.writerow([round(x,3), round(y,3)])
    return np.array(pts) if pts else np.empty((0,2))

def maybe_topology_lines(world_map):
    lines = []
    try:
        topo = world_map.get_topology()  # list of (wp_start, wp_end)
        for a, b in topo:
            ax, ay = a.transform.location.x, a.transform.location.y
            bx, by = b.transform.location.x, b.transform.location.y
            lines.append(((ax, ay), (bx, by)))
    except Exception:
        pass
    return lines

def main():
    client = carla.Client("localhost", 2000)
    world, world_map = ensure_map(client, EXPECTED_MAP)
    map_tag = world_map.name.split("/")[-1]  # e.g., Town03

    # Region for ped sampling
    if REGION_OVERRIDE:
        minx = REGION_OVERRIDE["minx"]; maxx = REGION_OVERRIDE["maxx"]
        miny = REGION_OVERRIDE["miny"]; maxy = REGION_OVERRIDE["maxy"]
    else:
        dx0, dx1, dy0, dy1 = driving_bbox(world_map)
        minx = dx0 - PED_PAD_M; maxx = dx1 + PED_PAD_M
        miny = dy0 - PED_PAD_M; maxy = dy1 + PED_PAD_M

    # Export CSVs
    veh_csv = os.path.join(OUTPUT_DIR, f"vehicle_spawns_{map_tag}.csv")
    ped_csv = os.path.join(OUTPUT_DIR, f"ped_points_{map_tag}.csv")
    spawns = export_vehicle_spawns(world_map, veh_csv)
    ped_pts = export_ped_samples(world_map, (minx, maxx, miny, maxy), PED_STEP_M, ped_csv)

    print(f"[ok] Wrote vehicle spawns: {veh_csv}  (count={len(spawns)})")
    print(f"[ok] Wrote pedestrian points: {ped_csv}  (count={len(ped_pts)})")

    # Prepare plot
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    ax.set_aspect('equal', adjustable='box')

    # Light road graph in background (optional)
    if DRAW_TOPOLOGY:
        for (ax0, ay0), (bx0, by0) in maybe_topology_lines(world_map):
            ax.plot([ax0, bx0], [ay0, by0], linewidth=0.5, alpha=0.25)

    # Plot ped points (magenta)
    if ped_pts.size:
        ax.scatter(ped_pts[:,0], ped_pts[:,1], s=2, alpha=0.6, label="Ped-walkable (sampled)")

    # Plot vehicle spawns (green) and label them
    vx = [tf.location.x for tf in spawns]
    vy = [tf.location.y for tf in spawns]
    ax.scatter(vx, vy, s=12, label="Vehicle spawns")
    for i, tf in enumerate(spawns):
        ax.text(tf.location.x, tf.location.y, f"V{i}", fontsize=6, ha="left", va="bottom")

    ax.set_title(f"{map_tag}: Vehicle Spawns + Ped-Walkable Samples")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(loc="best")
    # Nice bounds
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)

    png = os.path.join(OUTPUT_DIR, f"spawn_plot_{map_tag}.png")
    svg = os.path.join(OUTPUT_DIR, f"spawn_plot_{map_tag}.svg")
    fig.tight_layout()
    fig.savefig(png)
    fig.savefig(svg)
    print(f"[ok] Wrote plot: {png}")
    print(f"[ok] Wrote plot: {svg} (vector)")

if __name__ == "__main__":
    main()
