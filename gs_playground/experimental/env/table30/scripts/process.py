#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JSONL Processor:
- For each JSON line, apply:
  1) z += 0.1525
  2) yaw_new = -pi/2 - yaw_old   (radians)
- Write processed JSONL to output_dir with SAME filenames.

Supports common schemas:
- ee_pose: [x, y, z, roll, pitch, yaw]
- flat: ee_z / ee_yaw
- nested dot-path candidates (editable below)

Usage:
  python jsonl_apply_z_yaw_transform.py --input_dir /path/to/in --output_dir /path/to/out --max_files 10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


Z_OFFSET = 0.1525


# ------------------------- helpers: dot-path get/set -------------------------

def get_by_path(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def set_by_path(d: Dict[str, Any], path: str, value: Any) -> bool:
    cur: Any = d
    keys = path.split(".")
    for k in keys[:-1]:
        if not isinstance(cur, dict) or k not in cur:
            return False
        cur = cur[k]
    last = keys[-1]
    if isinstance(cur, dict) and last in cur:
        cur[last] = value
        return True
    return False


def wrap_to_pi(angle: float) -> float:
    # optional: wrap to (-pi, pi]
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ------------------------- core per-row transform -------------------------

def try_update_ee_pose_list(pose: Any, wrap_yaw: bool) -> Optional[List[float]]:
    """
    pose expected: [x, y, z, roll, pitch, yaw]
    returns updated list if valid, else None
    """
    if not isinstance(pose, list) or len(pose) < 6:
        return None
    try:
        x, y, z, r, p, yaw = pose[:6]
        z = float(z) + Z_OFFSET
        yaw_new = (-np.pi / 2.0) - float(yaw)
        if wrap_yaw:
            yaw_new = wrap_to_pi(yaw_new)
        out = list(pose)
        out[2] = z
        out[5] = yaw_new
        return out
    except Exception:
        return None


def transform_row(row: Dict[str, Any], wrap_yaw: bool = False) -> bool:
    """
    Mutates row in place.
    Returns True if any update was applied.
    """
    updated = False

    # 1) flat ee_pose
    if "ee_pose" in row:
        new_pose = try_update_ee_pose_list(row.get("ee_pose"), wrap_yaw)
        if new_pose is not None:
            row["ee_pose"] = new_pose
            updated = True

    # 2) flat columns ee_z / ee_yaw
    if "ee_z" in row:
        try:
            row["ee_z"] = float(row["ee_z"]) + Z_OFFSET
            updated = True
        except Exception:
            pass

    if "ee_yaw" in row:
        try:
            yaw_new = (-np.pi / 2.0) - float(row["ee_yaw"])
            if wrap_yaw:
                yaw_new = wrap_to_pi(yaw_new)
            row["ee_yaw"] = yaw_new
            updated = True
        except Exception:
            pass

    # 3) nested candidates (edit as needed)
    nested_pose_candidates = [
        "data.robots.Franka_1.arms.arm.ee_pose",
        "data.robots.franka_1.arms.arm.ee_pose",
        # 如果你的真机日志里用的是别的路径，把它们加在这里
    ]

    for path in nested_pose_candidates:
        v = get_by_path(row, path)
        if v is None:
            continue
        new_pose = try_update_ee_pose_list(v, wrap_yaw)
        if new_pose is not None:
            if set_by_path(row, path, new_pose):
                updated = True

    return updated


# ------------------------- IO -------------------------

def process_file(in_path: str, out_path: str, wrap_yaw: bool = False) -> Dict[str, int]:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n_in = 0
    n_out = 0
    n_updated = 0

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            s = line.strip()
            if not s:
                continue
            n_in += 1
            try:
                row = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {in_path}:{line_no}: {e}") from e

            if isinstance(row, dict):
                if transform_row(row, wrap_yaw=wrap_yaw):
                    n_updated += 1

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    return {"lines_in": n_in, "lines_out": n_out, "lines_updated": n_updated}


def run(input_dir: str, output_dir: str, max_files: int = 10, wrap_yaw: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in: {input_dir}")

    files = files[: max(1, min(int(max_files), 10))]

    print(f"[INFO] processing {len(files)} file(s)")
    for fp in files:
        name = os.path.basename(fp)
        out_fp = os.path.join(output_dir, name)
        stats = process_file(fp, out_fp, wrap_yaw=wrap_yaw)
        print(f"[OK] {name}: {stats}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder containing .jsonl files")
    ap.add_argument("--output_dir", required=True, help="Output folder")
    ap.add_argument("--max_files", type=int, default=1001, help="Max number of jsonl files to process (1..10)")
    ap.add_argument("--wrap_yaw", action="store_true", help="Optionally wrap yaw to (-pi, pi]")
    args = ap.parse_args()

    run(args.input_dir, args.output_dir, max_files=args.max_files, wrap_yaw=args.wrap_yaw)


if __name__ == "__main__":
    main()
