import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from matplotlib.lines import Line2D

# --------- Helpers ---------

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
    # Cast time-like columns to numeric
    for col in [
        "sim_time", "v1_brake_time", "v2_brake_time",
        "v1_stop_time", "v2_stop_time", "v2v_emit_time", "v2v_recv_time"
    ]:
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

def infer_mode_from_df(df: pd.DataFrame) -> Optional[str]:
    if "trigger_reason" not in df.columns:
        return None
    tr = df["trigger_reason"].dropna()
    if len(tr) == 0:
        return None
    mode = str(tr.iloc[-1]).strip().lower()
    if mode in ("perception", "vision"):
        return "perception"
    if mode in ("v2v", "v2x"):
        return "v2v"
    if mode == "hybrid":
        return "hybrid"
    return mode

def pick_one_by_policy(candidates, policy="first"):
    if not candidates:
        return None
    if policy == "first":
        return candidates[0]
    if policy == "latest":
        candidates = sorted(candidates, key=lambda x: os.path.getmtime(x["path"]), reverse=True)
        return candidates[0]
    if policy == "best":
        # Choose the one with smallest reaction latency
        candidates = [c for c in candidates if c.get("reaction_latency") is not None]
        if not candidates:
            return None
        candidates = sorted(candidates, key=lambda x: x["reaction_latency"])
        return candidates[0]
    return candidates[0]

# --------- Main plotting ---------

def plot_gap_vs_time(
    root: str = ".",
    out: str = "gap_vs_time",
    smooth: int = 0,
    dpi: int = 150,
    pick: str = "first",
):
    csvs = discover_csvs(root)
    if not csvs:
        raise SystemExit("No CSV files found in the current directory.")

    # Collect candidates by mode
    by_mode = {"perception": [], "v2v": [], "hybrid": []}
    for path in csvs:
        try:
            df = pd.read_csv(path)
            if "sim_time" not in df.columns or "gap_m" not in df.columns:
                continue
            mode = infer_mode_from_df(df)
            if mode not in by_mode:
                continue
            ev = summarize_events(df)
            by_mode[mode].append({"path": path, "events": ev, "reaction_latency": ev.get("reaction_latency")})
        except Exception:
            continue

    # Pick one per mode according to policy
    selection = {}
    for mode in ["perception", "v2v", "hybrid"]:
        chosen = pick_one_by_policy(by_mode[mode], policy=pick)
        if chosen is not None:
            selection[mode] = chosen

    if not selection:
        raise SystemExit("Found CSVs, but none with recognizable trigger_reason for perception/v2v/hybrid.")

    # Prepare plot
    plt.figure(figsize=(8, 4.5))

    color_map = {
        "perception": "#1f77b4",  # blue
        "v2v":        "#d62728",  # red
        "hybrid":     "#2ca02c",  # green
    }
    label_map = {
        "perception": "Perception",
        "v2v": "V2V",
        "hybrid": "Hybrid",
    }

    # Marker shapes (we'll build a separate legend with shape-only handles)
    marker_style = {
        "v1_brake_time": dict(marker="o",  label="V1 brake"),
        "v2_brake_time": dict(marker="^",  label="V2 brake"),
        "v1_stop_time":  dict(marker="s",  label="V1 stop"),
        "v2_stop_time":  dict(marker="*",  label="V2 stop"),
    }

    for mode, meta in selection.items():
        path = meta["path"]
        df = pd.read_csv(path).sort_values("sim_time")

        x = pd.to_numeric(df["sim_time"], errors="coerce")
        # ABSOLUTE GAP for the curve
        y = pd.to_numeric(df["gap_m"], errors="coerce").abs()
        if smooth and smooth > 1:
            y = y.rolling(window=smooth, min_periods=1, center=True).median()

        plt.plot(x, y, label=label_map.get(mode, mode), color=color_map.get(mode), linewidth=2)

        # Event markers (also ABS absolute value to match curve)
        ev = meta["events"]
        for ev_key, style in marker_style.items():
            t = ev.get(ev_key)
            if t is None:
                continue
            gap_at_t = nearest_value_at(df, t, "gap_m")
            if gap_at_t is None:
                continue
            gap_at_t = abs(gap_at_t)
            plt.scatter([t], [gap_at_t], color=color_map.get(mode), s=60, marker=style["marker"])

    # Legends: one for line colors (modes), one for marker shapes (events)
    line_handles = [
        Line2D([0], [0], color=color_map[m], lw=2, label=label_map[m])
        for m in selection.keys()
    ]
    marker_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="k", label="V1 brake"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="k", label="V2 brake"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="k", label="V1 stop"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="k", label="V2 stop"),
    ]

    legend1 = plt.legend(handles=line_handles, title="Mode", loc="upper right")
    plt.gca().add_artist(legend1)
    plt.legend(handles=marker_handles, title="Events", loc="lower right")

    plt.xlabel("Simulation time (s)")
    plt.ylabel("Gap (m)")
    plt.title("Gap vs. Time across Modes")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    png_path = f"{out}.png"
    pdf_path = f"{out}.pdf"
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {png_path} and {pdf_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot gap vs sim_time for Perception, V2V, Hybrid from CADET logs.")
    parser.add_argument("--out", type=str, default="gap_vs_time", help="Output figure basename (no extension)")
    parser.add_argument("--smooth", type=int, default=0, help="Rolling median window for smoothing (0 or 1 = no smoothing)")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI for PNG")
    parser.add_argument("--pick", type=str, choices=["first","latest","best"], default="first",
                        help="If multiple runs per mode exist: 'first' file, 'latest' by mtime, or 'best' (min reaction latency)")
    args = parser.parse_args()
    plot_gap_vs_time(root=".", out=args.out, smooth=args.smooth, dpi=args.dpi, pick=args.pick)
