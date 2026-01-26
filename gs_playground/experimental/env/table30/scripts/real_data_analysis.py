#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Franka JSONL Analyzer (real-robot friendly)

- Quaternion is read as wxyz: [qw, qx, qy, qz]
- Euler computed by SciPy Rotation:
    R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=False)

Compatible with:
- Single .jsonl file OR a folder of .jsonl files
- Very short sequences (1–4 samples) without crashing

Outputs:
  output_dir/summary.csv
  output_dir/selected.csv
  output_dir/overall_2x2_ee_xyz_gripper.png
  output_dir/overall_euler_xyz_deg.png
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Tuple, Union, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.spatial.transform import Rotation as R
except Exception as e:
    raise ImportError(
        "This script requires SciPy: `pip install scipy`"
    ) from e


# ----------------------------- Utilities -----------------------------

def read_jsonl_rows(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
    return rows


def safe_numeric_array(values: List[Any]) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def list_jsonl_files(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        return sorted(glob.glob(os.path.join(input_path, "*.jsonl")))
    if os.path.isfile(input_path) and input_path.lower().endswith(".jsonl"):
        return [input_path]
    raise FileNotFoundError(f"Input must be a folder or a .jsonl file: {input_path}")


def nanmin(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.min()) if a.size else float("nan")


def nanmax(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.max()) if a.size else float("nan")


def nrange(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.max() - a.min()) if a.size else float("nan")


def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    out = np.array(a, dtype=float, copy=True)
    m = np.isfinite(out)
    out[m] = (out[m] + math.pi) % (2 * math.pi) - math.pi
    return out


def rad2deg(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=float) * (180.0 / math.pi)


# ----------------------------- Robust key access -----------------------------

def _get_by_path(d: dict, path: str) -> Any:
    """
    Get nested value by dot path, e.g. "data.robots.Franka_1.arms.arm.ee_positions"
    Returns None if not found.
    """
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def get_timestamp(row: dict) -> Optional[float]:
    # flat
    if "timestamp" in row:
        try:
            return float(row["timestamp"])
        except Exception:
            return None
    # nested variants (add more if needed)
    candidates = [
        "data.timestamp",
        "data.time",
        "header.stamp",
    ]
    for c in candidates:
        v = _get_by_path(row, c)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return None


def get_ee_positions(row: dict) -> Optional[List[float]]:
    """
    Return ee_positions as [x,y,z,qw,qx,qy,qz] (wxyz quaternion).
    Supports both flat and nested schemas.
    """
    # flat schema
    if isinstance(row.get("ee_positions", None), list):
        return row["ee_positions"]

    # common real-robot / middleware nested keys (based on your screenshot)
    candidates = [
        "data.robots.Franka_1.arms.arm.ee_positions",
        "data.robots.franka_1.arms.arm.ee_positions",
        "data.robots.Franka_1.arms.arm.ee_pose",
        "data.robots.franka_1.arms.arm.ee_pose",
    ]
    for c in candidates:
        v = _get_by_path(row, c)
        if isinstance(v, list):
            return v

    return None


def get_gripper_value(row: dict) -> float:
    """
    Prefer:
      - gripper_width[0]
    Else:
      - sum(gripper)
    Else:
      - try nested common keys
    """
    if isinstance(row.get("gripper_width", None), list) and len(row["gripper_width"]) > 0:
        try:
            return float(row["gripper_width"][0])
        except Exception:
            return float("nan")

    if isinstance(row.get("gripper", None), list) and len(row["gripper"]) > 0:
        try:
            return float(np.nansum(np.array(row["gripper"], dtype=float)))
        except Exception:
            return float("nan")

    # nested guesses (extend if your real log uses other keys)
    nested_candidates = [
        "data.robots.Franka_1.gripper.width",
        "data.robots.franka_1.gripper.width",
        "data.robots.Franka_1.gripper.gripper_width",
        "data.robots.franka_1.gripper.gripper_width",
    ]
    for c in nested_candidates:
        v = _get_by_path(row, c)
        if isinstance(v, list) and len(v) > 0:
            try:
                return float(v[0])
            except Exception:
                return float("nan")
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass

    return float("nan")


# ----------------------------- Extraction -----------------------------

def extract_series_from_rows(rows: List[dict]) -> Dict[str, np.ndarray]:
    n = len(rows)
    if n == 0:
        return {k: np.array([], dtype=float) for k in ["t","x","y","z","qw","qx","qy","qz","g"]}

    # time: prefer timestamp, else index
    ts_vals: List[Any] = [get_timestamp(r) for r in rows]
    ts = safe_numeric_array(ts_vals)
    if np.isfinite(ts).sum() >= 2:
        t0 = ts[np.isfinite(ts)][0]
        t = ts - t0
    else:
        t = np.arange(n, dtype=float)

    # ee_positions: [x,y,z,qw,qx,qy,qz]  (wxyz)
    ee_list: List[List[float]] = []
    for r in rows:
        v = get_ee_positions(r)
        if isinstance(v, list):
            ee_list.append(v[:7] + [np.nan] * max(0, 7 - len(v)))
        else:
            ee_list.append([np.nan]*7)

    ee = np.array(ee_list, dtype=float)

    x, y, z = ee[:, 0], ee[:, 1], ee[:, 2]
    qw, qx, qy, qz = ee[:, 3], ee[:, 4], ee[:, 5], ee[:, 6]

    # gripper
    g = safe_numeric_array([get_gripper_value(r) for r in rows])

    return {"t": t, "x": x, "y": y, "z": z, "qw": qw, "qx": qx, "qy": qy, "qz": qz, "g": g}


# ----------------------------- Quaternion(wxyz) -> Euler(xyz) via SciPy -----------------------------

def quat_wxyz_to_euler_xyz(
    qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray,
    degrees: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Input quaternion arrays: wxyz (qw,qx,qy,qz)
    SciPy expects xyzw, so we reorder to (qx,qy,qz,qw).
    Returns (ex, ey, ez) in radians by default (degrees=False).
    """
    qw = np.asarray(qw, dtype=float)
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    qz = np.asarray(qz, dtype=float)

    ex = np.full_like(qw, np.nan)
    ey = np.full_like(qw, np.nan)
    ez = np.full_like(qw, np.nan)

    m = np.isfinite(qw) & np.isfinite(qx) & np.isfinite(qy) & np.isfinite(qz)
    if m.sum() == 0:
        return ex, ey, ez

    q = np.stack([ qw[m],qx[m], qy[m], qz[m]], axis=1)  # xyzw for SciPy

    # normalize to be safe (real logs sometimes drift slightly)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    good = np.isfinite(norm[:, 0]) & (norm[:, 0] > 0)
    if good.sum() == 0:
        return ex, ey, ez

    qn = q[good] / norm[good]
    e = R.from_quat(qn).as_euler("xyz", degrees=degrees)

    idx = np.where(m)[0][good]
    ex[idx], ey[idx], ez[idx] = e[:, 0], e[:, 1], e[:, 2]
    return ex, ey, ez


# ----------------------------- Summary / Selection -----------------------------

def summarize_one_file(path: str) -> Dict[str, Union[str, float]]:
    rows = read_jsonl_rows(path)
    name = os.path.basename(path)
    s = extract_series_from_rows(rows)

    x, y, z, g = s["x"], s["y"], s["z"], s["g"]
    x_min, x_max = nanmin(x), nanmax(x)
    y_min, y_max = nanmin(y), nanmax(y)
    z_min, z_max = nanmin(z), nanmax(z)

    pos_max = np.nanmax([x_max, y_max, z_max])
    neg_min = np.nanmin([x_min, y_min, z_min])

    return {
        "file": name,
        "n_steps": float(len(rows)),
        "pos_max": float(pos_max) if np.isfinite(pos_max) else float("nan"),
        "neg_min": float(neg_min) if np.isfinite(neg_min) else float("nan"),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "dx": nrange(x),
        "dy": nrange(y),
        "dz": nrange(z),
        "g_min": nanmin(g),
        "g_max": nanmax(g),
    }


def select_files_for_plotting(summary_df: pd.DataFrame) -> pd.DataFrame:
    if len(summary_df) <= 20:
        return summary_df.reset_index(drop=True)

    df = summary_df.copy()
    df_pos = df.dropna(subset=["pos_max"]).sort_values("pos_max", ascending=False)
    df_neg = df.dropna(subset=["neg_min"]).sort_values("neg_min", ascending=True)

    picked = pd.concat([df_pos.head(10), df_neg.head(10)], axis=0).drop_duplicates(subset=["file"], keep="first")

    if len(picked) < 20:
        remaining = df_pos[~df_pos["file"].isin(picked["file"])]
        picked = pd.concat([picked, remaining.head(20 - len(picked))], axis=0).drop_duplicates(subset=["file"], keep="first")

    if len(picked) < 20:
        remaining = df_neg[~df_neg["file"].isin(picked["file"])]
        picked = pd.concat([picked, remaining.head(20 - len(picked))], axis=0).drop_duplicates(subset=["file"], keep="first")

    return picked.head(20).reset_index(drop=True)


# ----------------------------- Plotting -----------------------------

def _plot_line(ax, t: np.ndarray, y: np.ndarray, label: str) -> bool:
    m = np.isfinite(t) & np.isfinite(y)
    if m.sum() >= 2:
        ax.plot(t[m], y[m], linewidth=1, label=label)
        return True
    return False


def plot_overall_2x2(series_list: List[Dict[str, np.ndarray]], labels: List[str], outpath: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    ax_x, ax_y = axes[0, 0], axes[0, 1]
    ax_z, ax_g = axes[1, 0], axes[1, 1]

    any_drawn = False
    for s, lb in zip(series_list, labels):
        any_drawn |= _plot_line(ax_x, s["t"], s["x"], lb)
        any_drawn |= _plot_line(ax_y, s["t"], s["y"], lb)
        any_drawn |= _plot_line(ax_z, s["t"], s["z"], lb)
        any_drawn |= _plot_line(ax_g, s["t"], s["g"], lb)

    ax_x.set_title("EE X vs time")
    ax_y.set_title("EE Y vs time")
    ax_z.set_title("EE Z vs time")
    ax_g.set_title("Gripper vs time")

    for ax in [ax_x, ax_y, ax_z, ax_g]:
        ax.set_xlabel("time (s)")

    ax_x.set_ylabel("x")
    ax_y.set_ylabel("y")
    ax_z.set_ylabel("z")
    ax_g.set_ylabel("gripper")

    ax_x.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    if any_drawn:
        fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_overall_euler_xyz_deg(series_list: List[Dict[str, np.ndarray]], labels: List[str], outpath: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    ax_x, ax_y, ax_z = axes

    any_drawn = False
    for s, lb in zip(series_list, labels):
        ex, ey, ez = quat_wxyz_to_euler_xyz(s["qw"], s["qx"], s["qy"], s["qz"], degrees=False)
        exd = rad2deg(wrap_to_pi(ex))
        eyd = rad2deg(wrap_to_pi(ey))
        ezd = rad2deg(wrap_to_pi(ez))

        any_drawn |= _plot_line(ax_x, s["t"], exd, lb)
        any_drawn |= _plot_line(ax_y, s["t"], eyd, lb)
        any_drawn |= _plot_line(ax_z, s["t"], ezd, lb)

    ax_x.set_title("Euler X vs time (degrees)  [SciPy as_euler('xyz')]")
    ax_y.set_title("Euler Y vs time (degrees)  [SciPy as_euler('xyz')]")
    ax_z.set_title("Euler Z vs time (degrees)  [SciPy as_euler('xyz')]")

    for ax in [ax_x, ax_y, ax_z]:
        ax.set_xlabel("time (s)")
        ax.set_ylabel("deg")

    ax_x.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    if any_drawn:
        fig.savefig(outpath, dpi=180)
    plt.close(fig)


# ----------------------------- Runner -----------------------------

def run(input_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    files = list_jsonl_files(input_path)
    if not files:
        raise FileNotFoundError(f"No .jsonl files found under: {input_path}")

    summaries: List[Dict[str, Union[str, float]]] = [summarize_one_file(fp) for fp in files]
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False, encoding="utf-8")

    selected_df = select_files_for_plotting(summary_df)
    selected_df.to_csv(os.path.join(output_dir, "selected.csv"), index=False, encoding="utf-8")

    base_dir = input_path if os.path.isdir(input_path) else os.path.dirname(os.path.abspath(input_path))

    series_list: List[Dict[str, np.ndarray]] = []
    labels: List[str] = []

    for fname in selected_df["file"].tolist():
        fp = os.path.join(base_dir, fname)
        if not os.path.exists(fp):
            if os.path.isfile(input_path) and os.path.basename(input_path) == fname:
                fp = input_path
            else:
                continue

        rows = read_jsonl_rows(fp)
        s = extract_series_from_rows(rows)
        series_list.append(s)
        labels.append(os.path.splitext(os.path.basename(fp))[0])

    if series_list:
        plot_overall_2x2(series_list, labels, os.path.join(output_dir, "overall_2x2_ee_xyz_gripper.png"))
        plot_overall_euler_xyz_deg(series_list, labels, os.path.join(output_dir, "overall_euler_xyz_deg.png"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder of .jsonl files OR a single .jsonl file")
    ap.add_argument("--output_dir", required=True, help="Output folder")
    args = ap.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
