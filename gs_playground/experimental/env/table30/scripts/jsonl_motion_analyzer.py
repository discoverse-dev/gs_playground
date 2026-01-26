#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JSONL Folder Analyzer + Combined Overlay Figures (<=10 files)

What you requested (implemented):
1) Read all .jsonl files in a folder (process up to 10 files, sorted).
2) Produce a combined 2x2 overlay figure (ALL files on each subplot, x-axis is time):
   - EE X vs t
   - EE Y vs t
   - EE Z vs t
   - Gripper vs t
3) Produce an additional combined angle figure (ALL files overlaid, x-axis is time):
   - Roll/Pitch/Yaw vs t (degrees; input angles are radians in [-pi, pi])
4) Write a summary.csv for quick numeric ranges.

Usage:
  python jsonl_motion_analyzer.py --input_dir /path/to/jsonl_folder --output_dir /path/to/output --max_files 10

Outputs:
  output_dir/summary.csv
  output_dir/overall_2x2_ee_xyz_gripper.png
  output_dir/overall_rpy_deg.png

Assumptions:
- Angles are radians already limited to [-pi, pi].
- Time is taken from the first available column among: t, time, timestamp, step; otherwise uses row index.

If your schema differs, edit _extract_ee_pose() / _extract_gripper() or key names.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- IO -----------------------------

def read_jsonl_to_df(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
    return pd.json_normalize(rows) if rows else pd.DataFrame()


def _as_float_array(v, length: int) -> List[float]:
    if isinstance(v, list):
        out = v[:length] + [np.nan] * max(0, length - len(v))
        return [float(x) if x is not None else np.nan for x in out]
    return [np.nan] * length


def _extract_time(df: pd.DataFrame) -> np.ndarray:
    # Prefer a real time column if present; else use index as t.
    for k in ["t", "time", "timestamp", "step"]:
        if k in df.columns:
            t = pd.to_numeric(df[k], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(t).sum() >= 2:
                return t
    return np.arange(len(df), dtype=float)


def _extract_ee_pose(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: x,y,z,roll,pitch,yaw arrays (NaNs if missing).
    Supports either:
      - ee_pose vector: [x, y, z, roll, pitch, yaw]
      - already-flat columns: ee_x/ee_roll/...
    """
    if "ee_pose" in df.columns:
        pose = df["ee_pose"].apply(lambda v: _as_float_array(v, 6))
        mat = np.vstack(pose.values) if len(pose) else np.empty((0, 6))
        x, y, z, r, p, yw = (mat[:, i] for i in range(6))
        return x, y, z, r, p, yw

    def col_or_nan(name: str) -> np.ndarray:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        return np.full(len(df), np.nan, dtype=float)

    x = col_or_nan("ee_x")
    y = col_or_nan("ee_y")
    z = col_or_nan("ee_z")
    r = col_or_nan("ee_roll")
    p = col_or_nan("ee_pitch")
    yw = col_or_nan("ee_yaw")
    return x, y, z, r, p, yw


def _extract_gripper(df: pd.DataFrame) -> np.ndarray:
    """
    Returns a 1D gripper array (NaNs if missing).
    Supports:
      - gripper vector [g]
      - scalar gripper
      - flat column gripper_0
    """
    if "gripper" in df.columns:
        def to_scalar(v):
            if isinstance(v, list) and len(v) > 0:
                return v[0]
            return v
        g = df["gripper"].apply(to_scalar)
        return pd.to_numeric(g, errors="coerce").to_numpy(dtype=float)

    if "gripper_0" in df.columns:
        return pd.to_numeric(df["gripper_0"], errors="coerce").to_numpy(dtype=float)

    return np.full(len(df), np.nan, dtype=float)


# ----------------------------- Analytics -----------------------------

def rad2deg(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=float) * (180.0 / np.pi)


def summarize_file(df: pd.DataFrame, filename: str) -> Dict[str, float]:
    t = _extract_time(df)
    x, y, z, r, p, yw = _extract_ee_pose(df)
    g = _extract_gripper(df)

    def rng(a: np.ndarray) -> float:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return float("nan")
        return float(a.max() - a.min())

    def mn(a: np.ndarray) -> float:
        a = a[np.isfinite(a)]
        return float(a.min()) if a.size else float("nan")

    def mx(a: np.ndarray) -> float:
        a = a[np.isfinite(a)]
        return float(a.max()) if a.size else float("nan")

    xyz = np.stack([x, y, z], axis=1)
    good = np.all(np.isfinite(xyz), axis=1)
    xyz2 = xyz[good]
    if xyz2.shape[0] >= 2:
        path_len = float(np.nansum(np.linalg.norm(np.diff(xyz2, axis=0), axis=1)))
    else:
        path_len = float("nan")

    out = {
        "file": filename,
        "n_steps": float(len(df)),
        "t_start": mn(t),
        "t_end": mx(t),
        "ee_x_range": rng(x),
        "ee_y_range": rng(y),
        "ee_z_range": rng(z),
        "ee_path_length": path_len,
        "gripper_min": mn(g),
        "gripper_max": mx(g),
        "roll_range_rad": rng(r),
        "pitch_range_rad": rng(p),
        "yaw_range_rad": rng(yw),
    }
    return out


# ----------------------------- Plotting -----------------------------

def plot_overall_2x2_overlay(
    series_list: List[Dict[str, np.ndarray]],
    labels: List[str],
    output_path: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    ax_x, ax_y = axes[0, 0], axes[0, 1]
    ax_z, ax_g = axes[1, 0], axes[1, 1]

    for s, lb in zip(series_list, labels):
        t = s["t"]
        x = s["x"]
        y = s["y"]
        z = s["z"]
        g = s["g"]

        mx = np.isfinite(t) & np.isfinite(x)
        my = np.isfinite(t) & np.isfinite(y)
        mz = np.isfinite(t) & np.isfinite(z)
        mg = np.isfinite(t) & np.isfinite(g)

        if mx.sum() >= 2:
            ax_x.plot(t[mx], x[mx], linewidth=1, label=lb)
        if my.sum() >= 2:
            ax_y.plot(t[my], y[my], linewidth=1, label=lb)
        if mz.sum() >= 2:
            ax_z.plot(t[mz], z[mz], linewidth=1, label=lb)
        if mg.sum() >= 2:
            ax_g.plot(t[mg], g[mg], linewidth=1, label=lb)

    ax_x.set_title("EE X vs time")
    ax_y.set_title("EE Y vs time")
    ax_z.set_title("EE Z vs time")
    ax_g.set_title("Gripper vs time")

    for ax in [ax_x, ax_y, ax_z, ax_g]:
        ax.set_xlabel("time (t)")

    ax_x.set_ylabel("EE X")
    ax_y.set_ylabel("EE Y")
    ax_z.set_ylabel("EE Z")
    ax_g.set_ylabel("Gripper")

    ax_x.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_overall_rpy_deg_overlay(
    series_list: List[Dict[str, np.ndarray]],
    labels: List[str],
    output_path: str,
) -> None:
    """
    Single figure for angles (degrees), x-axis is time.
    Overlay all files on each axis (roll/pitch/yaw).
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    ax_r, ax_p, ax_y = axes

    any_rpy = False

    for s, lb in zip(series_list, labels):
        t = s["t"]
        r_deg = rad2deg(s["roll"])
        p_deg = rad2deg(s["pitch"])
        y_deg = rad2deg(s["yaw"])

        mr = np.isfinite(t) & np.isfinite(r_deg)
        mp = np.isfinite(t) & np.isfinite(p_deg)
        my = np.isfinite(t) & np.isfinite(y_deg)

        if mr.sum() >= 2:
            ax_r.plot(t[mr], r_deg[mr], linewidth=1, label=lb)
            any_rpy = True
        if mp.sum() >= 2:
            ax_p.plot(t[mp], p_deg[mp], linewidth=1, label=lb)
            any_rpy = True
        if my.sum() >= 2:
            ax_y.plot(t[my], y_deg[my], linewidth=1, label=lb)
            any_rpy = True

    ax_r.set_title("Roll vs time (degrees)")
    ax_p.set_title("Pitch vs time (degrees)")
    ax_y.set_title("Yaw vs time (degrees)")

    for ax in [ax_r, ax_p, ax_y]:
        ax.set_xlabel("time (t)")
        ax.set_ylabel("deg")

    ax_r.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    if any_rpy:
        fig.savefig(output_path, dpi=180)
    plt.close(fig)


# ----------------------------- Main -----------------------------

def run(input_dir: str, output_dir: str, max_files: int = 10) -> None:
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in: {input_dir}")

    files = files[: max(1, min(max_files, 10))]

    summaries: List[Dict[str, float]] = []
    series_list: List[Dict[str, np.ndarray]] = []
    labels: List[str] = []

    for fp in files:
        name = os.path.basename(fp)
        df = read_jsonl_to_df(fp)
        if df.empty:
            continue

        t = _extract_time(df)
        x, y, z, r, p, yw = _extract_ee_pose(df)
        g = _extract_gripper(df)

        summaries.append(summarize_file(df, name))
        series_list.append({"t": t, "x": x, "y": y, "z": z, "g": g, "roll": r, "pitch": p, "yaw": yw})
        labels.append(os.path.splitext(name)[0])

    if summaries:
        pd.DataFrame(summaries).to_csv(os.path.join(output_dir, "summary.csv"), index=False, encoding="utf-8")

    if series_list:
        plot_overall_2x2_overlay(
            series_list=series_list,
            labels=labels,
            output_path=os.path.join(output_dir, "overall_2x2_ee_xyz_gripper.png"),
        )

        # angles in a separate figure
        plot_overall_rpy_deg_overlay(
            series_list=series_list,
            labels=labels,
            output_path=os.path.join(output_dir, "overall_rpy_deg.png"),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder containing .jsonl files")
    ap.add_argument("--output_dir", required=True, help="Output folder")
    ap.add_argument("--max_files", type=int, default=10, help="Max number of jsonl files to process (1..10)")
    args = ap.parse_args()

    max_files = max(1, min(int(args.max_files), 10))
    run(args.input_dir, args.output_dir, max_files=max_files)


if __name__ == "__main__":
    main()
