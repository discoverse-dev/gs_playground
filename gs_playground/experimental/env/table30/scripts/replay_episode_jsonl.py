#!/usr/bin/env python3
# replay_episode_jsonl.py
#
# Replay an episode JSONL by feeding each record's `ctrl` back into the env via env.step().
# Real-time render + save MP4 using OpenCV VideoWriter.
#
# CLI (only):
#   python replay_episode_jsonl.py -i /path/to/episode_00000.jsonl
#   python replay_episode_jsonl.py -i /path/to/episode_00000.jsonl -o /path/to/out.mp4
#   python replay_episode_jsonl.py -i /path/to/episode_00000.jsonl -o /path/to/output_dir/
#
# Quit: press 'q' in the window.

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2  # opencv-python

from gs_playground.src.manipulation.tasks.table30._02_stack_color_blocks_franka  import StackColorBlocksEnvCfg as FrankaCfg
from gs_playground.src.manipulation.tasks.table30._02_stack_color_blocks_franka  import StackColorBlocksEnv as FrankaEnv


def load_episode_jsonl(path: str) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                recs.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON decode failed at line {ln}: {e}") from e
    if not recs:
        raise RuntimeError(f"No records found in: {path}")
    return recs


def extract_ctrls(recs: List[Dict[str, Any]]) -> np.ndarray:
    ctrls = []
    for i, r in enumerate(recs):
        if "ctrl" not in r:
            raise KeyError(f"Missing 'ctrl' in record {i}")
        ctrls.append(r["ctrl"])
    arr = np.asarray(ctrls, dtype=np.float32)
    # expected (T, 7): [dx, dy, dz, droll, dpitch, dyaw, gripper]
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(f"Unexpected ctrl shape {arr.shape}, expected (T, 7)")
    return arr


def infer_dt(recs: List[Dict[str, Any]], default_dt: float = 0.02) -> float:
    """Infer dt from 'time' if present; else default."""
    if len(recs) < 2:
        return default_dt
    if "time" not in recs[0]:
        return default_dt
    try:
        ts = np.asarray([r["time"] for r in recs], dtype=np.float64)
        dts = np.diff(ts)
        dts = dts[np.isfinite(dts)]
        if dts.size == 0:
            return default_dt
        dt = float(np.median(dts))
        if dt <= 0.0 or dt > 1.0:
            return default_dt
        return dt
    except Exception:
        return default_dt


# --------------------------
# Env return-shape adapters
# --------------------------
def unwrap_reset(ret: Any) -> Dict[str, Any]:
    """
    Your TaskEnv.reset() returns (obs_dict, info_dict).
    Some wrappers might return RenderEnvState or just obs_dict.
    """
    # RenderEnvState-like
    if hasattr(ret, "obs") and isinstance(getattr(ret, "obs"), dict):
        return ret.obs  # type: ignore
    if isinstance(ret, (tuple, list)):
        return ret[0]
    if isinstance(ret, dict):
        return ret
    raise TypeError(f"Unsupported reset() return type: {type(ret)}")


def unpack_step(step_out: Any) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
    """
    Supports:
      - RenderEnvState (gs_playground)
      - Gymnasium: (obs, reward, terminated, truncated, info)
      - Gym:       (obs, reward, done, info)
      - Custom:    (obs, info)
    """
    # RenderEnvState-like
    if hasattr(step_out, "obs") and hasattr(step_out, "terminated") and hasattr(step_out, "truncated"):
        obs = step_out.obs  # type: ignore
        terminated = np.asarray(step_out.terminated).reshape(-1)  # type: ignore
        truncated = np.asarray(step_out.truncated).reshape(-1)  # type: ignore
        done = bool((terminated | truncated)[0]) if terminated.size > 0 else False
        info = step_out.info if hasattr(step_out, "info") else {}  # type: ignore
        return obs, done, info

    if not isinstance(step_out, (tuple, list)):
        raise TypeError(f"env.step returned {type(step_out)}")

    if len(step_out) == 5:
        obs, _reward, terminated, truncated, info = step_out

        def _as_bool(x: Any) -> bool:
            if isinstance(x, (bool, np.bool_)):
                return bool(x)
            x = np.asarray(x)
            return bool(x.reshape(-1)[0])

        done = _as_bool(terminated) or _as_bool(truncated)
        return obs, done, info

    if len(step_out) == 4:
        obs, _reward, done, info = step_out
        if not isinstance(done, (bool, np.bool_)):
            done = bool(np.asarray(done).reshape(-1)[0])
        return obs, bool(done), info

    if len(step_out) == 2:
        obs, info = step_out
        return obs, False, info

    raise ValueError(f"Unexpected env.step output length: {len(step_out)}")


# --------------------------
# Video helpers
# --------------------------
def infer_cam_key(obs: Dict[str, Any]) -> str:
    if "pixels/view_0" in obs:
        return "pixels/view_0"
    for k in obs.keys():
        if isinstance(k, str) and k.startswith("pixels/"):
            return k
    raise RuntimeError("No pixels/* key found in obs.")


def to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    f = np.asarray(frame, dtype=np.float32)
    if f.size > 0 and f.max() <= 1.5:
        f = f * 255.0
    return np.clip(f, 0.0, 255.0).astype(np.uint8)


def normalize_output_path(in_jsonl: str, out_arg: Optional[str]) -> str:
    base = os.path.splitext(os.path.basename(in_jsonl))[0]
    default_name = f"{base}_replay.mp4"

    if not out_arg:
        return os.path.join(os.path.dirname(in_jsonl), default_name)

    out_arg = os.path.expanduser(out_arg)

    # treat as directory if endswith / or is an existing directory
    if out_arg.endswith(os.sep) or out_arg.endswith("/"):
        return os.path.join(out_arg, default_name)
    if os.path.isdir(out_arg):
        return os.path.join(out_arg, default_name)

    # else file path
    return out_arg


def maybe_sync_reference_to_first_record(env: Any, first_rec: Dict[str, Any]) -> None:
    """
    eef_relative depends on robot.ref_ee_pose.
    Align it to the first record's ee_pose (if present) for tighter replay.
    """
    if "ee_pose" not in first_rec:
        return
    ee_pose = np.asarray(first_rec["ee_pose"], dtype=np.float32).reshape(-1)
    if ee_pose.shape[0] != 6:
        return

    # TaskEnv keeps internal state at env._state
    if not hasattr(env, "_state") or env._state is None:
        return
    data = env._state.data

    env.robot.update_reference(data)
    if getattr(env.robot, "ref_ee_pose", None) is not None:
        env.robot.ref_ee_pose[0] = ee_pose.copy()
    if getattr(env.robot, "last_cmd_ee_pose", None) is not None:
        env.robot.last_cmd_ee_pose[0] = ee_pose.copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, help="Path to episode_XXXXX.jsonl")
    parser.add_argument("-o", default=None, help="Output mp4 path OR output directory (optional)")
    args = parser.parse_args()

    recs = load_episode_jsonl(args.i)
    ctrls = extract_ctrls(recs)
    dt = infer_dt(recs, default_dt=0.02)
    fps = max(1, int(round(1.0 / dt)))

    out_path = normalize_output_path(args.i, args.o)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Build env (franka, eef_relative)
    cfg = FrankaCfg()
    try:
        cfg.action_mode = "eef_relative"
    except Exception:
        pass
    env = FrankaEnv(cfg, num_envs=1)

    # Reset (may return (obs, info) or state)
    obs = unwrap_reset(env.reset())

    # Ensure reference is initialized for relative control
    # (TaskEnv.reset calls robot.reset_envs internally; we still update_reference here)
    if hasattr(env, "_state") and env._state is not None:
        env.robot.update_reference(env._state.data)

    maybe_sync_reference_to_first_record(env, recs[0])

    cam_key = infer_cam_key(obs)

    # Init writer from first frame
    frame0_rgb = to_uint8_rgb(obs[cam_key][0])
    h, w = frame0_rgb.shape[:2]
    frame0_bgr = frame0_rgb[..., ::-1].copy()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter failed to open: {out_path}. "
            f"Try codec 'avc1'/'H264' if available, or check ffmpeg support in OpenCV."
        )

    win_name = "replay (press 'q' to quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    t0 = time.perf_counter()
    try:
        # write/show reset frame
        writer.write(frame0_bgr)
        cv2.imshow(win_name, frame0_bgr)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            return

        for i in range(ctrls.shape[0]):
            act = ctrls[i][None, :]  # (1,7)
            step_out = env.step(act)
            obs, done, _info = unpack_step(step_out)

            frame_rgb = to_uint8_rgb(obs[cam_key][0])
            frame_bgr = frame_rgb[..., ::-1].copy()

            writer.write(frame_bgr)
            cv2.imshow(win_name, frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            # realtime pacing
            target = (i + 1) * dt
            while True:
                now = time.perf_counter() - t0
                if now >= target:
                    break
                time.sleep(0.001)

            if done:
                break

    finally:
        writer.release()
        cv2.destroyAllWindows()

    print(f"Saved video: {out_path}")


if __name__ == "__main__":
    main()