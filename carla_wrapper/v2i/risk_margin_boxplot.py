#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- configurable knobs (no args needed) --------
PATTERN = "*latency_location_log.csv"
DEFAULT_PED_SPEED_MPS = 1.4  # fallback if ped_speed_mps is 0/NaN
ETA_EGO_MIN = 0.0            # seconds; discard zero/negatives
ETA_EGO_MAX = 12.0           # seconds; keep frames near decision horizon
WINSOR_LO, WINSOR_HI = 0.02, 0.98
OPTIONAL_REQUIRE_NEAR_START = False   # set True to require sim_time >= ped_start_time - LOOKBACK_S
LOOKBACK_S = 0.5
# -----------------------------------------------------

MODE_MAP     = {'v2i':'V2I', 'baseline':'Baseline', 'v2v':'V2V'}
PLATFORM_MAP = {'local':'Local', 'cloud':'Cloud', 'external':'Cloud', 'test':'Local-Test'}
OCCL_MAP     = {'none':'None', 'light':'Light', 'high':'High', 'test':'Test'}

ORDER = [
    'V2I-Local-None', 'V2I-Local-Light', 'V2I-Local-High',
    'V2I-Cloud-None', 'V2I-Cloud-Light', 'V2I-Cloud-High',
    'Baseline-Cloud-None', 'Baseline-Cloud-Light', 'Baseline-Cloud-High',
    'Baseline-Local-None', 'Baseline-Local-Light', 'Baseline-Local-High',
    'Baseline-Local-Test',
    'V2V-Local-None', 'V2V-Local-Light', 'V2V-Local-High',
    'V2V-Cloud-None', 'V2V-Cloud-Light', 'V2V-Cloud-High',
]

def parse_group_from_filename(fname: str) -> str:
    base = os.path.basename(fname).lower().replace('.csv','')
    toks = base.split('_')
    mode = next((MODE_MAP[t] for t in toks[:3] if t in MODE_MAP), 'Unknown')
    platform = next((PLATFORM_MAP[t] for t in toks[:4] if t in PLATFORM_MAP), 'Local')
    occl = next((OCCL_MAP[t] for t in toks[:5] if t in OCCL_MAP), 'None')
    if platform == 'Local-Test':
        platform, occl = 'Local', 'Test'
    return f"{mode}-{platform}-{occl}"

def _planar_eta(x_from, y_from, x_to, y_to, speed):
    dist = np.sqrt((x_to - x_from)**2 + (y_to - y_from)**2)
    v = pd.to_numeric(speed, errors='coerce').replace(0, np.nan)
    return dist / v

def compute_risk_margin(df: pd.DataFrame) -> pd.Series:
    req = ['x_ego','y_ego','ego_speed_mps_log','crossing_x','crossing_y',
           'p_x','p_y','ped_speed_mps','ped_started','ped_start_time','sim_time']
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"missing columns: {miss}")

    # ETA_ego (planar)
    eta_ego = _planar_eta(df['x_ego'], df['y_ego'], df['crossing_x'], df['crossing_y'], df['ego_speed_mps_log'])

    # ETA_ped (planar) + start delay if not started
    travel_time = _planar_eta(df['p_x'], df['p_y'], df['crossing_x'], df['crossing_y'],
                              df['ped_speed_mps'].where(pd.to_numeric(df['ped_speed_mps'], errors='coerce') > 0,
                                                        DEFAULT_PED_SPEED_MPS))
    delay = (pd.to_numeric(df['ped_start_time'], errors='coerce')
             - pd.to_numeric(df['sim_time'], errors='coerce')).clip(lower=0)
    ped_started = df['ped_started'].astype(bool)
    eta_ped = np.where(ped_started, travel_time, delay + travel_time)

    # Risk margin with your old sign convention: positive = safer gap
    rm = pd.to_numeric(eta_ped - eta_ego, errors='coerce').replace([np.inf, -np.inf], np.nan)

    # Keep decision-relevant frames
    keep = (eta_ego > ETA_EGO_MIN) & (eta_ego < ETA_EGO_MAX)
    if OPTIONAL_REQUIRE_NEAR_START:
        keep &= (pd.to_numeric(df['sim_time'], errors='coerce') >=
                 pd.to_numeric(df['ped_start_time'], errors='coerce') - LOOKBACK_S)
    rm = rm.where(keep)

    return rm.dropna()

def winsorize(s: pd.Series, lo=WINSOR_LO, hi=WINSOR_HI) -> pd.Series:
    if s.empty: return s
    ql, qh = s.quantile(lo), s.quantile(hi)
    return s.clip(lower=ql, upper=qh)

def main():
    files = sorted(glob.glob(PATTERN))
    if not files:
        print(f"No files matching {PATTERN} in {os.getcwd()}")
        return

    group_to_vals = {}
    for f in files:
        try:
            df = pd.read_csv(f)
            rm = compute_risk_margin(df)
        except Exception as e:
            print(f"[WARN] {os.path.basename(f)}: {e}")
            continue
        if rm.empty:
            print(f"[WARN] {os.path.basename(f)}: no usable RM after filtering")
            continue
        rm = winsorize(rm)
        group = parse_group_from_filename(f)
        group_to_vals.setdefault(group, []).extend(rm.tolist())

    if not group_to_vals:
        print("No valid data collected.")
        return

    labels = [g for g in ORDER if g in group_to_vals] + [g for g in group_to_vals if g not in ORDER]
    data = [group_to_vals[g] for g in labels]

    plt.figure(figsize=(20, 6))
    bp = plt.boxplot(
        data,
        notch=False,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color='black', linewidth=1.3),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        boxprops=dict(linewidth=1.0, facecolor='#ff6b6b', alpha=0.5)
    )
    for w in bp['whiskers']: w.set_color('black')

    plt.xticks(range(1, len(labels)+1), labels, rotation=35, ha='right')
    plt.ylabel("Risk Margin (s)")
    plt.title("Distribution of Risk Margin (Box Plots)")
    plt.grid(axis='y', linestyle='--', alpha=0.35)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
