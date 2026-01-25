#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JSONL Folder Analyzer + Selection of 20 Episodes by Spatial Generalization

Your requirements (implemented):
1) Read ALL .jsonl files in a folder.
2) For each file, compute EE position ranges:
     dx = max(ee_x)-min(ee_x), dy, dz
   and also signed extremes:
     pos_max = max over axes of max(value)    (largest positive reach)
     neg_min = min over axes of min(value)    (largest negative reach; most negative)
3) Select:
   - Top 10 files with largest positive reach (pos_max DESC)
   - Top 10 files with largest negative reach magnitude (neg_min ASC, i.e., most negative)
   Union them as plotting set (target 20; if overlap, fill from next best to reach 20).
4) Plot (using ONLY these selected files):
   A) One combined 2x2 overlay figure (time on x-axis):
      - EE X vs t, EE Y vs t, EE Z vs t, Gripper vs t
   B) One combined angle figure (time on x-axis):
      - Roll/Pitch/Yaw vs t in DEGREES (input angles radians in [-pi, pi])
5) Write summary.csv for ALL files and selected.csv for the chosen 20.

Usage:
  python jsonl_motion_analyzer.py --input_dir /path/to/jsonl_folder --output_dir /path/to/output

Outputs:
  output_dir/summary.csv          (all files)
  output_dir/selected.csv         (the 20 selected for plotting)
  output_dir/overall_2x2_ee_xyz_gripper.png
  output_dir/overall_rpy_deg.png

Schema assumptions:
- Per JSON line, either:
  - ee_pose: [x, y, z, roll, pitch, yaw]   (recommended)
  - or flat columns: ee_x, ee_y, ee_z, ee_roll, ee_pitch, ee_yaw
- Gripper:
  - gripper: [g] or scalar
  - or gripper_0

Time axis:
- Uses first available column among: t, time, timestamp, step
- Otherwise uses row index as time.

Notes:
- This script DOES NOT unwrap angles; it directly converts rad->deg for display.
  If you want unwrap, ask and I will add it (keeping deg output).
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
    for k in ["t", "time", "timestamp", "step"]:
        if k in df.columns:
            t = pd.to_numeric(df[k], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(t).sum() >= 2:
                return t
    return np.arange(len(df), dtype=float)


def _extract_ee_pose(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns x,y,z,roll,pitch,yaw arrays (NaNs if missing).
    Supports ee_pose vector or flat columns.
    """
    if "ee_pose" in df.columns:
        pose = df["ee_pose"].apply(lambda v: _as_float_array(v, 6))
        if len(pose) == 0:
            n = len(df)
            return (np.full(n, np.nan),) * 6
        mat = np.vstack(pose.values)
        x, y, z, r, p, yw = (mat[:, i] for i in range(6))
        return x, y, z, r, p, yw

    def col_or_nan(name: str) -> np.ndarray:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        return np.full(len(df), np.nan, dtype=float)

    return (
        col_or_nan("ee_x"),
        col_or_nan("ee_y"),
        col_or_nan("ee_z"),
        col_or_nan("ee_roll"),
        col_or_nan("ee_pitch"),
        col_or_nan("ee_yaw"),
    )


def _extract_gripper(df: pd.DataFrame) -> np.ndarray:
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


def _nanmin(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.min()) if a.size else float("nan")


def _nanmax(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.max()) if a.size else float("nan")


def _range(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.max() - a.min()) if a.size else float("nan")


def summarize_file(path: str) -> Dict[str, float]:
    df = read_jsonl_to_df(path)
    name = os.path.basename(path)

    if df.empty:
        return {
            "file": name,
            "n_steps": 0.0,
            "pos_max": float("nan"),
            "neg_min": float("nan"),
            "ee_x_min": float("nan"),
            "ee_x_max": float("nan"),
            "ee_y_min": float("nan"),
            "ee_y_max": float("nan"),
            "ee_z_min": float("nan"),
            "ee_z_max": float("nan"),
            "ee_x_range": float("nan"),
            "ee_y_range": float("nan"),
            "ee_z_range": float("nan"),
        }

    x, y, z, r, p, yw = _extract_ee_pose(df)

    x_min, x_max = _nanmin(x), _nanmax(x)
    y_min, y_max = _nanmin(y), _nanmax(y)
    z_min, z_max = _nanmin(z), _nanmax(z)

    # Positive reach: largest positive coordinate across x/y/z
    pos_max = np.nanmax([x_max, y_max, z_max])
    # Negative reach: most negative coordinate across x/y/z
    neg_min = np.nanmin([x_min, y_min, z_min])

    return {
        "file": name,
        "n_steps": float(len(df)),
        "pos_max": float(pos_max) if np.isfinite(pos_max) else float("nan"),
        "neg_min": float(neg_min) if np.isfinite(neg_min) else float("nan"),
        "ee_x_min": float(x_min),
        "ee_x_max": float(x_max),
        "ee_y_min": float(y_min),
        "ee_y_max": float(y_max),
        "ee_z_min": float(z_min),
        "ee_z_max": float(z_max),
        "ee_x_range": _range(x),
        "ee_y_range": _range(y),
        "ee_z_range": _range(z),
        "roll_range_rad": _range(r),
        "pitch_range_rad": _range(p),
        "yaw_range_rad": _range(yw),
    }


def select_top_20(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select 20 files:
      - top 10 by pos_max descending
      - top 10 by neg_min ascending (most negative)
    If duplicates occur, fill from next ranks to reach 20.
    """
    df = summary_df.copy()

    # Drop rows with no pos/neg info
    df_pos = df.dropna(subset=["pos_max"]).sort_values("pos_max", ascending=False)
    df_neg = df.dropna(subset=["neg_min"]).sort_values("neg_min", ascending=True)

    pos_pick = df_pos.head(10)
    neg_pick = df_neg.head(10)

    picked = pd.concat([pos_pick, neg_pick], axis=0)

    # Deduplicate, then fill to 20 using remaining ranks (first from pos, then from neg)
    picked = picked.drop_duplicates(subset=["file"], keep="first")

    if len(picked) < 20:
        remaining = df_pos[~df_pos["file"].isin(picked["file"])].copy()
        need = 20 - len(picked)
        picked = pd.concat([picked, remaining.head(need)], axis=0).drop_duplicates(subset=["file"], keep="first")

    if len(picked) < 20:
        remaining = df_neg[~df_neg["file"].isin(picked["file"])].copy()
        need = 20 - len(picked)
        picked = pd.concat([picked, remaining.head(need)], axis=0).drop_duplicates(subset=["file"], keep="first")

    # Final clamp (in case there are fewer than 20 valid files total)
    return picked.head(20).reset_index(drop=True)


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

    ax_x.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_overall_rpy_deg_overlay(
    series_list: List[Dict[str, np.ndarray]],
    labels: List[str],
    output_path: str,
) -> None:
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

    ax_r.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    if any_rpy:
        fig.savefig(output_path, dpi=180)
    plt.close(fig)


# ----------------------------- Main -----------------------------

def run(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in: {input_dir}")

    # 1) Summarize ALL files (no limit)
    summaries: List[Dict[str, float]] = []
    for fp in files:
        summaries.append(summarize_file(fp))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False, encoding="utf-8")

    # 2) Select 20: top-10 positive reach + top-10 negative reach
    selected_df = select_top_20(summary_df)
    selected_df.to_csv(os.path.join(output_dir, "selected.csv"), index=False, encoding="utf-8")

    # 3) Load ONLY selected 20 for plotting
    selected_files = selected_df["file"].tolist()
    series_list: List[Dict[str, np.ndarray]] = []
    labels: List[str] = []

    for fname in selected_files:
        fp = os.path.join(input_dir, fname)
        if not os.path.exists(fp):
            # If path mismatch (rare), skip
            continue
        df = read_jsonl_to_df(fp)
        if df.empty:
            continue
        t = _extract_time(df)
        x, y, z, r, p, yw = _extract_ee_pose(df)
        g = _extract_gripper(df)

        series_list.append({"t": t, "x": x, "y": y, "z": z, "g": g, "roll": r, "pitch": p, "yaw": yw})
        labels.append(os.path.splitext(fname)[0])

    # 4) Plot figures using ONLY those 20
    if series_list:
        plot_overall_2x2_overlay(
            series_list=series_list,
            labels=labels,
            output_path=os.path.join(output_dir, "overall_2x2_ee_xyz_gripper.png"),
        )

        plot_overall_rpy_deg_overlay(
            series_list=series_list,
            labels=labels,
            output_path=os.path.join(output_dir, "overall_rpy_deg.png"),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder containing .jsonl files")
    ap.add_argument("--output_dir", required=True, help="Output folder")
    args = ap.parse_args()

    run(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
