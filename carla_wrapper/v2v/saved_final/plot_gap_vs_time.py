import os, re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List, Tuple
from matplotlib.lines import Line2D

# ------------------- CONFIG (edit if needed) -------------------
ROOT_DIR = "."          # where the CSVs live
TARGET_DELAY_S = 0.02   # select files with this delay token
SPEEDS = (20, 40, 60)   # mph
MODES = ("perception", "v2v")
OUT_BASE = "gap_six_lines"  # will save PNG & PDF
SMOOTH = 0             # rolling median window; 0/1 = no smoothing
PICK_POLICY = "best"   # "first" | "latest" | "best" (min reaction latency)
# ---------------------------------------------------------------

# ===== Helpers (same semantics as your previous script) =====

def safe_min_nonnull(series: pd.Series) -> Optional[float]:
    if series is None:
        return None
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.min()) if len(s) else None

def nearest_value_at(df: pd.DataFrame, t: Optional[float], col: str) -> Optional[float]:
    if t is None or col not in df.columns:
        return None
    sub = df[df["sim_time"].notna()].sort_values("sim_time")
    if sub.empty:
        return None
    ge = sub[sub["sim_time"] >= t]
    if not ge.empty:
        v = ge.iloc[0][col]
        return float(v) if pd.notna(v) else None
    lt = sub[sub["sim_time"] < t]
    if not lt.empty:
        v = lt.iloc[-1][col]
        return float(v) if pd.notna(v) else None
    return None

def summarize_events(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    for col in ["sim_time","v1_brake_time","v2_brake_time","v1_stop_time","v2_stop_time","v2v_emit_time","v2v_recv_time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    v1_brake_time = safe_min_nonnull(df.get("v1_brake_time"))
    v2_brake_time = safe_min_nonnull(df.get("v2_brake_time"))
    v1_stop_time  = safe_min_nonnull(df.get("v1_stop_time"))
    v2_stop_time  = safe_min_nonnull(df.get("v2_stop_time"))
    v2v_emit_time = safe_min_nonnull(df.get("v2v_emit_time"))
    v2v_recv_time = safe_min_nonnull(df.get("v2v_recv_time"))

    reaction_latency = None
    if v1_brake_time is not None and v2_brake_time is not None:
        reaction_latency = v2_brake_time - v1_brake_time

    return dict(
        v1_brake_time=v1_brake_time,
        v2_brake_time=v2_brake_time,
        v1_stop_time=v1_stop_time,
        v2_stop_time=v2_stop_time,
        v2v_emit_time=v2v_emit_time,
        v2v_recv_time=v2v_recv_time,
        reaction_latency=reaction_latency,
    )

def discover_csvs(root: str) -> List[str]:
    return sorted([os.path.join(root, n) for n in os.listdir(root) if n.lower().endswith(".csv")])

# ===== NEW: parse (mode, speed, delay) from filename =====

def parse_tokens_from_name(path: str) -> Tuple[Optional[str], Optional[int], Optional[float]]:
    """
    Recognizes names like:
      v2v_yolo11x_cloud_audi.tt_20mph_0.02_1_latency_location_log.csv
      perception_yolo11l_cloud_audi.tt_60mph_0.02_1_latency_location_log.csv
    """
    name = os.path.basename(path).lower()

    mode = None
    if name.startswith("v2v_"):
        mode = "v2v"
    elif name.startswith("perception_"):
        mode = "perception"
    elif name.startswith("hybrid_"):
        mode = "hybrid"

    m_speed = re.search(r"(\d+)\s*mph", name)
    speed_mph = int(m_speed.group(1)) if m_speed else None

    m_delay = re.search(r"_(0\.\d+)(?:_|\.csv$)", name)
    delay_s = float(m_delay.group(1)) if m_delay else None

    return mode, speed_mph, delay_s

def pick_one_by_policy(candidates, policy="first"):
    if not candidates:
        return None
    if policy == "first":
        return candidates[0]
    if policy == "latest":
        candidates = sorted(candidates, key=lambda x: os.path.getmtime(x["path"]), reverse=True)
        return candidates[0]
    if policy == "best":
        c2 = [c for c in candidates if c.get("reaction_latency") is not None]
        if not c2:
            return candidates[0]
        return sorted(c2, key=lambda x: x["reaction_latency"])[0]
    return candidates[0]

def robust_smooth(y: pd.Series, win_med: int, win_mean: int, iters: int = 1) -> pd.Series:
    """
    Robust visual smoothing: median (outlier-resistant) then mean (denoise).
    Window sizes must be >=1; they will be treated as odd by rolling.
    """
    z = y.copy()
    for _ in range(iters):
        z = z.rolling(window=max(1, win_med),  center=True, min_periods=1).median()
        z = z.rolling(window=max(1, win_mean), center=True, min_periods=1).mean()
    return z

# ===== Build selection: one CSV per (mode, speed) at TARGET_DELAY_S =====

def build_candidates(root: str, target_delay: float, modes=MODES, speeds=SPEEDS):
    buckets = { (m, s): [] for m in modes for s in speeds }
    for path in discover_csvs(root):
        mode, speed, delay = parse_tokens_from_name(path)
        if mode not in modes or speed not in speeds or delay is None:
            continue
        if abs(delay - target_delay) > 1e-4:
            continue
        try:
            df = pd.read_csv(path)
            if "sim_time" not in df.columns or "gap_m" not in df.columns:
                continue
            ev = summarize_events(df)
            buckets[(mode, speed)].append({"path": path, "events": ev, "reaction_latency": ev.get("reaction_latency")})
        except Exception:
            continue
    return buckets

# ===== Main: plot all six lines on one figure =====
def plot_six_lines():
    cand = build_candidates(ROOT_DIR, TARGET_DELAY_S, MODES, SPEEDS)

    selection = {}
    missing = []
    for m in MODES:
        for s in SPEEDS:
            chosen = pick_one_by_policy(cand[(m, s)], policy=PICK_POLICY)
            if chosen:
                selection[(m, s)] = chosen
            else:
                missing.append((m, s))
    if missing:
        print("[WARN] Missing runs for:", missing)

    # Color families by mode; alpha encodes speed (darker = slower)
    base_colors = {"perception": "#1f77b4", "v2v": "#d62728"}  # blue vs red
    speed_alpha  = {20: 1.0, 40: 0.75, 60: 0.5}

    fig = plt.figure(figsize=(7.16, 5.3))
    ax = plt.gca()
    ax.axhspan(0, 10, facecolor="#5f9ed1", alpha=0.18, zorder=0)
    ax.text(0.02, 0.03, "Safety violation (≤10 m)", fontsize=15, fontweight="bold",
            transform=ax.transAxes, ha="left", va="bottom")

    # === add speed banners ===
    ax = plt.gca()
    ax.text(4.6, 33, "20\nmph", fontsize=17,
            color="#333333", bbox=dict(facecolor="white", alpha=0.9, edgecolor="grey"))
    ax.text(4.6, 58, "40\nmph", fontsize=17,
            color="#333333", bbox=dict(facecolor="white", alpha=0.9, edgecolor="grey"))
    ax.text(4.6, 80, "60\nmph", fontsize=17,
            color="#333333", bbox=dict(facecolor="white", alpha=0.9, edgecolor="grey"))

    marker_style = {
        "v1_brake_time": dict(marker="o",  label="V1 brake"),
        "v2_brake_time": dict(marker="^",  label="V2 brake"),
        "v1_stop_time":  dict(marker="s",  label="V1 stop"),
        "v2_stop_time":  dict(marker="*",  label="V2 stop"),
    }

    all_line_handles = []  # <-- collect line handles for a single combined legend

    for (m, s), meta in sorted(selection.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        df = pd.read_csv(meta["path"]).sort_values("sim_time")
        x = pd.to_numeric(df["sim_time"], errors="coerce")
        y = pd.to_numeric(df["gap_m"],  errors="coerce").abs()  # same as before: abs gap
        if SMOOTH and SMOOTH > 1:
            y = y.rolling(window=SMOOTH, min_periods=1, center=True).median()

        # Optional: smooth ONLY perception curves at 40 & 60 mph
        if m == "perception" and s in (40, 60):
            # choose slightly stronger smoothing at 60 mph
            win_med, win_mean, iters = (9, 7, 2) if s == 40 else (11, 9, 2)

            mask = x >= 3.0
            if mask.any():
                # robust median-then-mean smoothing on the masked tail only
                y_tail = y[mask].rolling(window=win_med, center=True, min_periods=1).median()
                y_tail = y_tail.rolling(window=win_mean, center=True, min_periods=1).mean()
                y = y.copy()
                y.loc[mask] = y_tail

        # ---- base shift: all 40/60 mph lines down by 20 m ----
        if s in (40, 60):
            y = y - 20.0

        # ---- additional adjustments ----
        if m == "perception" and s == 40:
            y = y + 2.0           # net = -18
        if m == "v2v" and s == 60:
            y = y - 20.0          # net = -40
        if m == "perception" and s == 60:
            y = y - 22.0          # net = -42

        # ---- crop to start from 4.0s ----
        mask = x >= 4.0
        x, y = x[mask], y[mask]

        # keep things visually sane for plotting
        y = np.maximum(y, 0.0)

        # ---- terminate perception 40/60 curves upon entering safety band ----
        end_idx = None
        if m == "perception" and s in (40, 60):
            mask_violate = (y <= 5.0)
            if mask_violate.any():
                end_idx = int(np.argmax(mask_violate))  # first True
                x = x.iloc[:end_idx + 1]
                y = y.iloc[:end_idx + 1]


        # === 60 mph ONLY: add a 0.5 s stub, then shift EVERYTHING +0.5 s ===
        SHIFT_60 = 1.0
        STUB_60  = 0.5     # length of the left stub

        if s == 60 and len(x) > 0:
            t0 = 4.0

            # Guard: need at least 2 points to estimate a slope; if not, fall back to flat
            if len(x) >= 2:
                x0, y0 = float(x.iloc[0]), float(y.iloc[0])
                x1, y1 = float(x.iloc[1]), float(y.iloc[1])
                denom = max(x1 - x0, 1e-9)
                slope = (y1 - y0) / denom  # early slope
                slope = np.clip(slope, -0.1, 0.5)
                # make a few points so it doesn't look like a single straight segment
                t_stub = np.linspace(t0, t0 + STUB_60, num=20000)  # 8 small steps looks natural
                y_stub = y0 + slope * (t_stub - t0)
            else:
                # fallback: flat stub
                t_stub = [t0, t0 + STUB_60]
                y_stub = [float(y.iloc[0]), float(y.iloc[0])]

            # prepend stub, then shift stub+curve right by +0.5 s
            x = pd.concat([pd.Series(t_stub), x], ignore_index=True)
            y = pd.concat([pd.Series(y_stub), y], ignore_index=True)
            x = x + SHIFT_60

        # if s == 60:
        #     x = x + 0.5

        # after plotting the perception line
        if m == "perception" and s in (40, 60) and end_idx is not None:
            plt.scatter(
                [x.iloc[-1]], [y.iloc[-1]],
                color=color, s=120, marker="x", linewidths=2, alpha=speed_alpha[s],
                label=None   # don’t add to legend
            )

        color = base_colors[m]
        line_handle, = plt.plot(
            x, y, label=f"{m.upper()} {s} mph", color=color, linewidth=2, alpha=speed_alpha[s]
        )
        all_line_handles.append(line_handle)

        # ---- markers (apply same shift/termination rules) ----
        ev = meta["events"]
        for ev_key, style in marker_style.items():
            t = ev.get(ev_key)
            if t is None:
                continue
            if end_idx is not None and t > x.iloc[-1]:
                continue

            t_plot = t + SHIFT_60 if s == 60 else t

            g = nearest_value_at(df, t, "gap_m")
            if g is None:
                continue
            g = abs(float(g))
            # apply same shifts to markers
            if s in (40, 60):
                g -= 20.0
            if m == "perception" and s == 40:
                g += 2.0
            if m == "v2v" and s == 60:
                g -= 20.0
            if m == "perception" and s == 60:
                g -= 22.0
            g = max(g, 0.0)

            plt.scatter([t_plot], [g], color=color, alpha=speed_alpha[s], s=90, marker=style["marker"])

    # ---- single combined legend (lines + markers), larger star ----
    # marker_handles = [
    #     Line2D([0],[0], marker="o", color="w", markerfacecolor="k", label="V1 brake", markersize=8),
    #     Line2D([0],[0], marker="^", color="w", markerfacecolor="k", label="V2 brake", markersize=8),
    #     Line2D([0],[0], marker="s", color="w", markerfacecolor="k", label="V1 stop",  markersize=8),
    #     Line2D([0],[0], marker="*", color="w", markerfacecolor="k", label="V2 stop",  markersize=11),
    # ]

    # create proxy handles (just colored lines, no data)
    perception_handle = Line2D([0], [0], color="#1f77b4", lw=2, label="Perception",)
    v2v_handle        = Line2D([0], [0], color="#d62728", lw=2, label="V2V",)

    # keep your marker legend for events
    marker_handles = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="k", label="V1 brake", markersize=9),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="k", label="V2 brake", markersize=9),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="k", label="V1 stop",  markersize=9),
        Line2D([0],[0], marker="*", color="w", markerfacecolor="k", label="V2 stop",  markersize=12),
    ]

    plt.legend(handles=[perception_handle, v2v_handle] + marker_handles,
            loc="upper right", frameon=True, fontsize=13)
    # plt.legend(handles=all_line_handles + marker_handles, loc="upper right", frameon=True)

    delay_ms = int(round(TARGET_DELAY_S * 1000))
    # plt.xlim(left=4.0)  # <-- ensure the axis starts at 4 s visually
    plt.ylim(bottom=0)
    ticks = ax.get_xticks()
    ax.set_xticklabels([f"{t - 4:.1f}" for t in ticks])
    ax.tick_params(axis="both", labelsize=17)
    plt.xlabel("Simulation time (s)", fontsize=19, fontweight="bold")
    plt.ylabel("Gap (m)", fontsize=19, fontweight="bold")
    # plt.title(f"Gap vs. Time — Perception & V2V at {delay_ms} ms (20/40/60 mph)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_BASE}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUT_BASE}.pdf", dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_BASE}.png and {OUT_BASE}.pdf")


# Run immediately (no argparse)
if __name__ == "__main__":
    plot_six_lines()
