#!/usr/bin/env python3
import os, re, glob, argparse
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------
# Helpers (kept from your file)
# ----------------------------
def ci_lookup(cols, *cands):
    if cols is None:
        return None
    try:
        n = len(cols)
    except Exception:
        n = 0
    if n == 0:
        return None
    cl = {str(c).lower(): c for c in cols}
    for cand in cands:
        if not cand:
            continue
        k = str(cand).lower()
        if k in cl:
            return cl[k]
    for c in cols:
        lc = str(c).lower()
        for cand in cands:
            if cand and str(cand).lower() in lc:
                return c
    return None

def coerce_num(s):
    return pd.to_numeric(s, errors="coerce")

def parse_client_meta(filename):
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    parts = name.split("_")
    mode_tok = parts[0] if parts else ""
    family = "V2I" if "v2i" in mode_tok.lower() else "Baseline"

    occlusion = "none"
    for tok in parts[:6]:
        t = tok.lower()
        if t in ("none","light","heavy"):
            occlusion = t; break

    speed = None
    m = re.search(r"(\d+)\s*mph", name, re.IGNORECASE)
    if m: speed = f"{m.group(1)}mph"

    txt = "_".join(parts).lower()
    site = "Local" if "local" in txt else ("Cloud" if ("cloud" in txt or "external" in txt) else "Local")
    return {"family": family, "site": site, "occlusion": occlusion, "speed": speed}

def label_from_family_site(fam, site): return f"{fam}-{site}"

# --------------------------------------
# Client-only risk series from one CSV
# --------------------------------------
def series_from_client_csv(path, units="seconds", debug=False):
    """
    Returns DataFrame with columns: time_s, risk_margin
    RM = TTC_e - TTE_p (GT) computed from CLIENT columns only.

    TTC_e uses ego (x,y) -> (crossing_x, crossing_y) and ego_speed_mps_capture (fallback to _log).
    TTE_p uses |crossing_x - p_x| / |v_px| if moving toward crossing (X-axis jaywalk).
    If units == "meters", multiplies RM_s by ego speed (m/s) row-wise.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        if debug: print(f"[DEBUG] failed to read {path}: {e}")
        return pd.DataFrame(columns=["time_s","risk_margin"])

    if debug: print(f"[DEBUG] client cols for {os.path.basename(path)}: {list(df.columns)}")

    # Look up columns (case-insensitive, tolerant to name drift)
    tcol = ci_lookup(df.columns, "sim_time", "time_s", "timestamp", "ts_s")
    ex   = ci_lookup(df.columns, "x_ego", "ego_x")
    ey   = ci_lookup(df.columns, "y_ego", "ego_y")
    cx   = ci_lookup(df.columns, "crossing_x", "x_cross", "cross_x")
    cy   = ci_lookup(df.columns, "crossing_y", "y_cross", "cross_y")

    # Ego speed (prefer capture-time; fall back to log-time; last resort any 'speed_mps')
    sp_cap = ci_lookup(df.columns, "ego_speed_mps_capture")
    sp_log = ci_lookup(df.columns, "ego_speed_mps_log")
    sp_any = ci_lookup(df.columns, "speed_mps")  # legacy
    spcol  = sp_cap or sp_log or sp_any

    # Pedestrian GT pose/vel (X axis)
    px   = ci_lookup(df.columns, "p_x", "px")
    vpx  = ci_lookup(df.columns, "v_px", "vpx")

    # Basic presence checks
    needed = [tcol, ex, ey, cx, cy, spcol, px, vpx]
    if any(c is None for c in needed):
        if debug:
            print("[DEBUG] missing columns:",
                  dict(time=tcol, x_ego=ex, y_ego=ey, crossing_x=cx, crossing_y=cy,
                       speed=spcol, p_x=px, v_px=vpx))
        return pd.DataFrame(columns=["time_s","risk_margin"])

    # Coerce numerics
    t  = coerce_num(df[tcol])
    exv = coerce_num(df[ex]); eyv = coerce_num(df[ey])
    cxv = coerce_num(df[cx]); cyv = coerce_num(df[cy])
    sp  = coerce_num(df[spcol])
    pxv = coerce_num(df[px]); vxv = coerce_num(df[vpx])

    # TTC_e = distance(ego -> crossing) / speed
    gap = np.sqrt((cxv - exv)**2 + (cyv - eyv)**2)
    ttc = gap / np.maximum(sp, 1e-6)
    ttc[~np.isfinite(ttc)] = np.nan

    # --- TTE_p (GT) along X toward crossing_x, ONLY after ped_started ---
    # ped_started column (bool-ish); fallback heuristic if missing
    ps_col = ci_lookup(df.columns, "ped_started", "ped_start_flag", "ped_moving")
    if ps_col is not None:
        ps = df[ps_col].astype(str).str.lower().isin(["true","1","yes"]) | (df[ps_col] == 1)
        ped_started_mask = ps.values
    else:
        # heuristic: consider "started" when |v_px| exceeds a small epsilon
        ped_started_mask = (np.abs(vxv) > 0.1).values if hasattr(vxv, "values") else (np.abs(vxv) > 0.1)

    dx = cxv - pxv
    moving_toward = ((vxv > 0) & (dx > 0)) | ((vxv < 0) & (dx < 0))
    valid_motion = (np.abs(vxv) > 1e-6) & moving_toward

    # only compute TTE when started AND valid_motion
    compute_mask = ped_started_mask & valid_motion.values
    tte = np.full_like(dx, np.nan, dtype=float)
    tte[compute_mask] = np.abs(dx[compute_mask]) / np.abs(vxv[compute_mask])

    # horizon clamp (optional, mirrors your server’s eval)
    tte[(tte > 6.0)] = np.nan

    # RM = TTC - TTE (seconds)
    rm_s = ttc - tte

    out = pd.DataFrame({"time_s": t, "risk_margin": rm_s}).dropna()
    if units.lower().startswith("m"):
        out["risk_margin"] = out["risk_margin"] * sp.loc[out.index]

    out = out[np.isfinite(out["time_s"]) & np.isfinite(out["risk_margin"])]
    out = out.sort_values("time_s").drop_duplicates(subset=["time_s"])
    if len(out) < 2:
        if debug: print(f"[DEBUG] client series too short after cleaning: {path}")
        return pd.DataFrame(columns=["time_s","risk_margin"])
    return out

def try_extract_markers_from_client(path, debug=False):
    """
    Finds: brake_received_s (first row with brake_issued True) and a heuristic full_stop_s
    based on ego_speed_mps_log/capture ~ 0 for several consecutive samples.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    # brake timestamp
    br_flag = ci_lookup(df.columns, "brake_issued", "brake", "brake_flag")
    tcol    = ci_lookup(df.columns, "sim_time", "time_s", "timestamp")
    res = {}
    if br_flag and tcol:
        mask = df[br_flag].astype(str).str.lower().isin(["true","1","yes"]) | (df[br_flag] == 1)
        ts = coerce_num(df.loc[mask, tcol]).dropna()
        if len(ts): res["brake_received_s"] = float(ts.iloc[0])

    # full stop heuristic
    sp_cap = ci_lookup(df.columns, "ego_speed_mps_capture")
    sp_log = ci_lookup(df.columns, "ego_speed_mps_log")
    scol = sp_log or sp_cap
    if scol and tcol:
        sp = coerce_num(df[scol]); tt = coerce_num(df[tcol])
        # consider stop when speed < 0.2 m/s for >= 0.5s window
        low = (sp.fillna(999.0) < 0.2).astype(int)
        # rolling sum over ~10 samples (if ~20 Hz); fallback simple last low
        win = max(5, min(30, len(low)//20 or 5))
        try:
            rs = low.rolling(window=win, min_periods=win).sum()
            idx = rs[rs == win].index
            if len(idx):
                res["full_stop_s"] = float(tt.loc[idx[0]])
        except Exception:
            if low.any():
                res["full_stop_s"] = float(tt.loc[low[low==1].index[-1]])
    return res

LINESTYLES = {"Baseline-Local":"-","Baseline-Cloud":"--","V2I-Local":":","V2I-Cloud":"-."}
OCCLORS = {"none":"#4C78A8","light":"#F58518","heavy":"#54A24B"}

# ----------------------------
# Main processing (client-only)
# ----------------------------
def process(root_dir=".", units="seconds", debug=False):
    client_logs = sorted(glob.glob(os.path.join(root_dir, "*latency_location_log.csv")))
    if debug:
        print("[DEBUG] found client logs:")
        for f in client_logs: print("   ", os.path.basename(f))
    groups = defaultdict(lambda: defaultdict(dict))

    # Build series per client file
    for cpath in client_logs:
        meta = parse_client_meta(os.path.basename(cpath))
        fam, site, occl, speed = meta["family"], meta["site"], meta["occlusion"], meta["speed"]
        label = label_from_family_site(fam, site)

        s = series_from_client_csv(cpath, units=units, debug=debug)
        if s.empty:
            print(f"[WARN] Client series empty after cleaning: {os.path.basename(cpath)} (skipping)")
            continue

        markers = try_extract_markers_from_client(cpath, debug=debug)
        groups[occl][speed][label] = {"series": s, "markers": markers, "client": cpath}
        if debug:
            print(f"[DEBUG] added client={os.path.basename(cpath)} as {label}")

    # Plot per occlusion/speed
    generated = []
    for occl, by_speed in groups.items():
        for speed, entries in by_speed.items():
            if not entries: continue
            all_times = np.unique(np.concatenate([v["series"]["time_s"].values for v in entries.values()]))
            if all_times.size == 0:
                print(f"[WARN] No times for occl={occl} speed={speed} (skipping).")
                continue
            interp = {}
            for lab, v in entries.items():
                s = v["series"]
                interp[lab] = np.interp(all_times, s["time_s"].values, s["risk_margin"].values)

            arr = np.vstack(list(interp.values()))
            lo = np.min(arr, axis=0); hi = np.max(arr, axis=0)

            fig, ax = plt.subplots(figsize=(10.8, 6.4))
            color = OCCLORS.get(occl, "#888")
            units_label = "s" if units.startswith("s") else "m"
            ax.fill_between(all_times, lo, hi, alpha=0.18, color=color, label=f"{occl.capitalize()} band")

            for lab in ["Baseline-Local","Baseline-Cloud","V2I-Local","V2I-Cloud"]:
                if lab in interp:
                    ax.plot(all_times, interp[lab], LINESTYLES[lab], linewidth=1.6, color=color, alpha=0.95, label=lab)

            for lab, v in entries.items():
                mk = v["markers"]
                if "brake_received_s" in mk: ax.axvline(mk["brake_received_s"], linestyle="--", linewidth=1.0, color=color, alpha=0.6)
                if "full_stop_s" in mk:      ax.axvline(mk["full_stop_s"], linestyle=":",  linewidth=1.0, color=color, alpha=0.6)

            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            leg1 = ax.legend(handles=[Patch(facecolor=color, edgecolor='none', alpha=0.18, label=f"{occl.capitalize()}")],
                             title="Occlusion Level (band = min↔max across configs)", loc="upper left")
            line_handles = [Line2D([0],[0], linestyle=LINESTYLES[l], color="black", linewidth=1.8, label=l)
                            for l in ["Baseline-Local","Baseline-Cloud","V2I-Local","V2I-Cloud"] if l in interp]
            leg2 = ax.legend(handles=line_handles, title="System Config (line style)", loc="lower right")
            ax.add_artist(leg1)

            ttl_speed = f" @ {speed}" if speed else ""
            ylab = f"Risk Margin (TTC−TTE) [{units_label}]"
            ax.set_title(f"Risk Margin Over Time{ttl_speed} — {occl.capitalize()} Occlusion")
            ax.set_xlabel("Time (s)"); ax.set_ylabel(ylab)
            ax.grid(True, linewidth=0.4, alpha=0.4)
            plt.tight_layout()
            out = f"risk_margin_{occl}{'_'+speed if speed else ''}_{units_label}.png"
            plt.savefig(out, dpi=300); plt.close(fig)
            generated.append(out)
            print(f"[OK] Wrote {out}")
    return generated

def main():
    ap = argparse.ArgumentParser(description="Build risk-margin figures from client CSVs only.")
    ap.add_argument("--root", default=".", help="Root directory to search (default: .)")
    ap.add_argument("--units", choices=["seconds","meters"], default="seconds",
                    help="Plot margin in seconds (TTC−TTE) or meters (margin_s * ego_speed_mps). Default: seconds")
    ap.add_argument("--debug", action="store_true", help="Verbose logs")
    args = ap.parse_args()
    outs = process(root_dir=args.root, units=args.units, debug=args.debug)
    if outs:
        print("\nGenerated files:")
        for p in outs: print(" -", p)

if __name__ == "__main__":
    main()
