# =============================================================================
# File: 02_stack_color_blocks_data_qpos_mediapy_dualview.py
# =============================================================================

from __future__ import annotations

import os
import json
import time
import shutil
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import mediapy as media
from scipy.spatial.transform import Rotation

from gs_playground.src.manipulation.tasks.table30._02_stack_color_blocks_franka import (
    StackColorBlocksEnv,
    StackColorBlocksEnvCfg,
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def smooth_step_pos(curr: np.ndarray, tgt: np.ndarray, max_dp: float) -> np.ndarray:
    dp = tgt - curr
    n = np.linalg.norm(dp, axis=1, keepdims=True)
    s = np.minimum(1.0, float(max_dp) / (n + 1e-9))
    return curr + dp * s


def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / (n + 1e-9)


# -----------------------------------------------------------------------------
# Video Writer (mediapy)
# -----------------------------------------------------------------------------
class EpisodeVideoWriter:
    """
    Stream mp4 writer using mediapy.VideoWriter.
    Expects RGB uint8 frames (H,W,3).
    """

    def __init__(
        self,
        path: str,
        fps: int,
        size_wh: Tuple[int, int],
        *,
        codec: str = "h264",
        qp: Optional[int] = None,
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        w, h = int(size_wh[0]), int(size_wh[1])
        self.shape_hw = (h, w)
        self._closed = False

        kwargs: Dict[str, Any] = {"codec": codec}
        if qp is not None:
            kwargs["qp"] = int(qp)

        self._ctx = media.VideoWriter(path, shape=self.shape_hw, fps=float(fps), **kwargs)
        self._writer = self._ctx.__enter__()

    def write(self, rgb: Optional[np.ndarray]) -> None:
        if self._closed or rgb is None:
            return

        img = np.asarray(rgb)

        # ensure uint8 RGB
        if img.dtype != np.uint8:
            img = media.to_uint8(img)

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected RGB (H,W,3), got {img.shape}")

        if tuple(img.shape[:2]) != tuple(self.shape_hw):
            img = media.resize_image(img, self.shape_hw)
            if img.dtype != np.uint8:
                img = media.to_uint8(img)

        self._writer.add_image(img)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._ctx.__exit__(None, None, None)
        finally:
            self._ctx = None
            self._writer = None


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CollectorCfg:
    # dataset
    data_size: int = 10
    num_envs: int = 5
    seed: int = 42
    save_dir: str = "./data/table30_stack_color_blocks_collect_full_manhattan_dualview_mediapy"

    # env control
    max_ctrl_steps: int = 800

    # motion params
    max_dp: float = 0.005
    pos_tol: float = 0.001

    # rotation control
    rot_gain: float = 0.6
    max_dr: float = 0.08
    yaw_tol: float = 0.03

    # optional yaw offsets
    yaw_offset_grasp: float = 0.0
    yaw_offset_place: float = 0.0

    # task offsets
    grasp_down_z: float = 0.0
    lift_dz: float = 0.06
    cube_half: float = 0.025

    # gripper
    gripper_open: float = 0.0
    gripper_close: float = 0.82

    # timing / dwell
    close_hold_steps: int = 15
    stack_hold_steps: int = 10
    waypoint_dwell_steps: int = 20

    # sampling / render
    sample_every_steps: int = 1
    save_video: bool = True
    render_every_steps: int = 1
    video_fps: int = 30
    video_w: int = 640
    video_h: int = 480

    # dual-view keys
    cam_view_key: Sequence[str] = field(default_factory=lambda: ["pixels/view_0", "pixels/view_1"])

    # JSONL pose fix
    ee_z_offset: float = 0.1525
    yaw_fix_bias: float = -0.5 * np.pi  # new_yaw = yaw_fix_bias - old_yaw

    # if obs ee_pose is 7D quaternion, which order?
    ee_quat_convention: str = "xyzw"  # "xyzw" | "wxyz"

    subtask: Optional[str] = "Stack specific colored blocks."


# -----------------------------------------------------------------------------
# Collector
# -----------------------------------------------------------------------------
class StackColorBlocksCollector:
    ST_APP_LIFT_Z = 0
    ST_APP_ALIGN_X = 1
    ST_APP_ALIGN_Y = 2
    ST_APP_ALIGN_YAW = 3
    ST_APP_DESCEND = 4

    ST_CLOSE = 5

    ST_TRP_LIFT_Z = 6
    ST_TRP_ALIGN_X = 7
    ST_TRP_ALIGN_Y = 8
    ST_TRP_ALIGN_YAW = 9
    ST_TO_STACK = 10

    ST_OPEN_HOLD = 11
    ST_RETREAT = 12
    ST_TO_HOME = 13
    ST_DONE = 14

    def __init__(self, cfg: CollectorCfg, env_cfg: Optional[StackColorBlocksEnvCfg] = None):
        self.cfg = cfg

        os.makedirs(cfg.save_dir, exist_ok=True)
        self.videos_dir = os.path.join(cfg.save_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

        self.env_cfg = env_cfg if env_cfg is not None else StackColorBlocksEnvCfg()
        self.env = StackColorBlocksEnv(self.env_cfg, num_envs=int(cfg.num_envs))
        self.env.reset()

        self.B = int(cfg.num_envs)

        self.cube_names = self.env_cfg.cube_names
        self.cube_bodies = self.env.cube_bodies

        self.cam_keys = list(cfg.cam_view_key)

        self.active = np.zeros(self.B, dtype=bool)
        self.done = np.zeros(self.B, dtype=bool)
        self.success = np.zeros(self.B, dtype=bool)
        self.ctrl_step = np.zeros(self.B, dtype=np.int32)

        self.states = np.zeros(self.B, dtype=np.int32)
        self.state_enter_step = np.zeros(self.B, dtype=np.int32)
        self.state_reach_step = np.full(self.B, -1, dtype=np.int32)

        self._attempt_id = np.zeros(self.B, dtype=np.int64)
        self.top_idx = np.zeros(self.B, dtype=np.int32)
        self.base_idx = np.zeros(self.B, dtype=np.int32)
        self.stack_hold_counter = np.zeros(self.B, dtype=np.int32)

        self.exec_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.exec_quat = np.zeros((self.B, 4), dtype=np.float32)

        self.latched_start_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.latched_top_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.latched_base_pos = np.zeros((self.B, 3), dtype=np.float32)

        self.latched_start_quat = np.zeros((self.B, 4), dtype=np.float32)
        self.latched_start_yaw = np.zeros((self.B,), dtype=np.float32)
        self.latched_top_yaw = np.zeros((self.B,), dtype=np.float32)
        self.latched_base_yaw = np.zeros((self.B,), dtype=np.float32)

        self.buffers: List[Dict[str, Any]] = [self._new_buffer() for _ in range(self.B)]

        self.video_writers: List[Dict[str, EpisodeVideoWriter]] = [{} for _ in range(self.B)]
        self._tmp_video_paths: List[Dict[str, str]] = []
        for i in range(self.B):
            per_cam = {}
            for key in self.cam_keys:
                safe_key = key.replace("/", "_")
                per_cam[key] = os.path.join(self.videos_dir, f"_tmp_env{i}_{safe_key}.mp4")
            self._tmp_video_paths.append(per_cam)

        self.saved_count = 0
        self.attempted = 0
        self._last_log_t = time.perf_counter()
        self._last_action = np.zeros((self.B, 7), dtype=np.float32)

    @staticmethod
    def _new_buffer() -> Dict[str, Any]:
        return {
            "times": [],
            "logic_states": [],
            "qpos": [],
            "ee_pose": [],
            "gripper": [],
            "ctrl": [],
            "reward": [],
            "top_idx": [],
            "base_idx": [],
            "is_success": [],
            "video_frames": 0,
            "frame_idxs": [],
        }

    # -----------------------------
    # Video
    # -----------------------------
    def _close_writers_for_env(self, env_id: int) -> None:
        writers = self.video_writers[env_id]
        for k in list(writers.keys()):
            try:
                writers[k].close()
            except Exception:
                pass
        self.video_writers[env_id] = {}

    def _reset_video_writer(self, env_id: int) -> None:
        self._close_writers_for_env(env_id)
        for key in self.cam_keys:
            tmp_path = self._tmp_video_paths[env_id][key]
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            self.video_writers[env_id][key] = EpisodeVideoWriter(
                tmp_path,
                int(self.cfg.video_fps),
                (int(self.cfg.video_w), int(self.cfg.video_h)),
            )

    def _write_video_frame(self, env_id: int) -> None:
        writers = self.video_writers[env_id]
        if not writers:
            return

        obs = self.env._state.obs
        wrote_any = False

        for key in self.cam_keys:
            if key in obs and key in writers:
                rgb = obs[key][env_id]
                if rgb is not None:
                    writers[key].write(rgb.copy())
                    wrote_any = True

        if wrote_any:
            self.buffers[env_id]["video_frames"] += 1

    # -----------------------------
    # Episode init/finalize
    # -----------------------------
    def start_episodes(self, env_ids: np.ndarray, seed: int) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if env_ids.size == 0:
            return

        try:
            self.env._rng = np.random.default_rng(int(seed))
        except Exception:
            pass

        done_mask = np.zeros((self.B,), dtype=bool)
        done_mask[env_ids] = True
        self.env.reset(done=done_mask)

        self.active[env_ids] = True
        self.done[env_ids] = False
        self.success[env_ids] = False
        self.ctrl_step[env_ids] = 0
        self.state_enter_step[env_ids] = 0
        self.stack_hold_counter[env_ids] = 0
        self.state_reach_step[env_ids] = -1

        data = self.env._state.data
        self.env.robot.reset_envs(data, done_mask)
        self.env.robot.update_reference(data)

        all_poses = self.env.robot.get_ee_pose(data)
        for idx in env_ids.tolist():
            pose = all_poses[idx]
            self.exec_pos[idx] = pose[:3].astype(np.float32)
            self.latched_start_pos[idx] = pose[:3].astype(np.float32)

            if len(pose) == 7:
                q = np.asarray(pose[3:7], dtype=np.float32)
                self.exec_quat[idx] = normalize_quat(q[None, :])[0]
            elif len(pose) == 6:
                euler = np.asarray(pose[3:6], dtype=np.float32)
                q = Rotation.from_euler("xyz", euler, degrees=False).as_quat().astype(np.float32)
                self.exec_quat[idx] = normalize_quat(q[None, :])[0]
            else:
                self.exec_quat[idx] = Rotation.identity().as_quat().astype(np.float32)

            self.latched_start_quat[idx] = self.exec_quat[idx].copy()
            self.latched_start_yaw[idx] = Rotation.from_quat(self.exec_quat[idx]).as_euler("xyz", degrees=False)[2].astype(np.float32)

        self.top_idx[env_ids] = self.env.top_idx[env_ids]
        self.base_idx[env_ids] = self.env.base_idx[env_ids]

        cube_pose = np.stack(
            [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.cube_bodies],
            axis=1,
        )

        self.latched_top_pos[env_ids] = cube_pose[env_ids, self.top_idx[env_ids], :3]
        self.latched_base_pos[env_ids] = cube_pose[env_ids, self.base_idx[env_ids], :3]

        top_q = cube_pose[env_ids, self.top_idx[env_ids], 3:7]
        base_q = cube_pose[env_ids, self.base_idx[env_ids], 3:7]
        top_yaw = Rotation.from_quat(top_q).as_euler("xyz", degrees=False)[:, 2]
        base_yaw = Rotation.from_quat(base_q).as_euler("xyz", degrees=False)[:, 2]
        self.latched_top_yaw[env_ids] = top_yaw.astype(np.float32) * -1
        self.latched_base_yaw[env_ids] = base_yaw.astype(np.float32) * -1

        self.states[env_ids] = self.ST_APP_LIFT_Z

        for env_id in env_ids.tolist():
            self.buffers[env_id] = self._new_buffer()
            if self.cfg.save_video:
                self._reset_video_writer(env_id)
            self._attempt_id[env_id] += 1
            self.attempted += 1

    def _finalize_episode(self, env_id: int) -> None:
        self._close_writers_for_env(env_id)

        if self.success[env_id] and self.saved_count < int(self.cfg.data_size):
            ep_idx = int(self.saved_count)
            saved_paths_map: Dict[str, str] = {}

            if self.cfg.save_video:
                for key in self.cam_keys:
                    tmp_path = self._tmp_video_paths[env_id][key]
                    if os.path.exists(tmp_path):
                        safe_key = key.replace("/", "_")
                        final_rel_path = f"videos/episode_{ep_idx:05d}_{safe_key}.mp4"
                        abs_path = os.path.join(self.cfg.save_dir, final_rel_path)
                        try:
                            shutil.move(tmp_path, abs_path)
                            saved_paths_map[key] = final_rel_path
                        except Exception as e:
                            print(f"[WARN] move video failed ({key}): {e}")

            self._flush_jsonl(env_id, ep_idx, saved_paths_map)
            self.saved_count += 1
            print(f"[Success] Saved episode {ep_idx}. Total saved: {self.saved_count}")

        for key in self.cam_keys:
            tmp_path = self._tmp_video_paths[env_id][key]
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        self.buffers[env_id] = self._new_buffer()

    # -----------------------------
    # Capture / JSONL (yaw+z fix)
    # -----------------------------
    def _capture_step(self, env_id: int) -> None:
        obs = self.env._state.obs
        buf = self.buffers[env_id]

        frame_idx = None
        if self.cfg.save_video:
            if (int(self.ctrl_step[env_id]) % int(self.cfg.render_every_steps)) == 0:
                frame_idx = int(buf["video_frames"])
        buf["frame_idxs"].append(frame_idx)

        buf["times"].append(float(self.ctrl_step[env_id] * 0.02))
        buf["logic_states"].append(int(self.states[env_id]))
        buf["qpos"].append(obs["qpos"][env_id].tolist())
        buf["ee_pose"].append(obs["ee_pose"][env_id].tolist())
        buf["gripper"].append(obs["gripper"][env_id].tolist())
        buf["ctrl"].append(self._last_action[env_id].tolist())

        is_success = bool(self.success[env_id])
        buf["is_success"].append(is_success)
        buf["reward"].append(float(1.0 if is_success else 0.0))

        buf["top_idx"].append(int(self.top_idx[env_id]))
        buf["base_idx"].append(int(self.base_idx[env_id]))

    def _ee_row_to_rpy(self, row: List[float]) -> Tuple[float, float, float]:
        arr = np.asarray(row, dtype=np.float32)
        if arr.shape[0] == 6:
            return float(arr[3]), float(arr[4]), float(arr[5])

        if arr.shape[0] == 7:
            q = arr[3:7].astype(np.float32)
            if str(self.cfg.ee_quat_convention).lower() == "wxyz":
                q = q[[1, 2, 3, 0]]  # -> xyzw
            q = normalize_quat(q[None, :])[0]
            rpy = Rotation.from_quat(q).as_euler("xyz", degrees=False)
            return float(rpy[0]), float(rpy[1]), float(rpy[2])

        return 0.0, 0.0, 0.0

    def _flush_jsonl(self, env_id: int, ep_idx: int, vid_paths_map: Dict[str, str]) -> None:
        path = os.path.join(self.cfg.save_dir, f"episode_{ep_idx:05d}.jsonl")
        buf = self.buffers[env_id]
        n = len(buf["times"])

        t_raw = self.cube_names[self.top_idx[env_id]]
        b_raw = self.cube_names[self.base_idx[env_id]]
        t_name = t_raw.replace("cube_", "").lower()
        b_name = b_raw.replace("cube_", "").lower()
        prompt = f"Stack the {t_name} block on top of the {b_name} block."

        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                raw_pose = buf["ee_pose"][i]
                raw_gripper = buf["gripper"][i]

                x, y, z = float(raw_pose[0]), float(raw_pose[1]), float(raw_pose[2])
                roll, pitch, yaw = self._ee_row_to_rpy(raw_pose)

                new_z = z + float(self.cfg.ee_z_offset)
                new_yaw = float(self.cfg.yaw_fix_bias) - float(yaw)
                new_yaw = float(wrap_to_pi(np.asarray(new_yaw, dtype=np.float32)))

                g_val = float(raw_gripper[0]) if isinstance(raw_gripper, (list, np.ndarray)) else float(raw_gripper)

                custom_ee_pose_7d = [x, y, new_z, roll, pitch, new_yaw, g_val]

                rec: Dict[str, Any] = {
                    "prompt": prompt,
                    "qpos": buf["qpos"][i],
                    "ee_pose": custom_ee_pose_7d,
                    "gripper": buf["gripper"][i],
                    "ctrl": buf["ctrl"][i],
                    "is_robot": True,
                    "top_idx": buf["top_idx"][i],
                    "base_idx": buf["base_idx"][i],
                }

                frame_idx = buf.get("frame_idxs", [None] * n)[i]
                if frame_idx is None:
                    frame_idx = i

                for k_idx, key in enumerate(self.cam_keys):
                    json_key = f"images_{k_idx + 1}"
                    if key in vid_paths_map:
                        rec[json_key] = {"url": vid_paths_map[key], "type": "video", "frame_idx": int(frame_idx)}

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # -----------------------------
    # Success check / FSM
    # -----------------------------
    def _check_stack_success(self, mask: np.ndarray) -> np.ndarray:
        data = self.env._state.data
        cube_pose = np.stack([np.asarray(b.get_pose(data), dtype=np.float32) for b in self.cube_bodies], axis=1)

        row_ids = np.arange(self.B)
        t_idx = self.top_idx
        b_idx = self.base_idx

        tp = cube_pose[row_ids, t_idx, :3]
        bp = cube_pose[row_ids, b_idx, :3]
        xy_dist = np.linalg.norm(tp[:, :2] - bp[:, :2], axis=1)
        z_diff = tp[:, 2] - bp[:, 2]
        target_z = 2.0 * float(self.cfg.cube_half)

        xy_ok = xy_dist < 0.03
        z_ok = np.abs(z_diff - target_z) < 0.02
        return (xy_ok & z_ok) & mask

    def _enter_state(self, mask: np.ndarray, new_state: int) -> None:
        if not np.any(mask):
            return
        self.states[mask] = int(new_state)
        self.state_enter_step[mask] = self.ctrl_step[mask].copy()
        self.state_reach_step[mask] = -1

    def _step_logic(self) -> None:
        cfg = self.cfg
        B = self.B

        running = self.active & (~self.done)
        if not np.any(running):
            return

        start_p = self.latched_start_pos
        top_p = self.latched_top_pos
        base_p = self.latched_base_pos

        safe_z = top_p[:, 2] + float(cfg.lift_dz)

        p_app_lift_z = start_p.copy()
        p_app_lift_z[:, 2] = safe_z

        p_app_align_x = p_app_lift_z.copy()
        p_app_align_x[:, 0] = top_p[:, 0]

        p_app_align_y = top_p.copy()
        p_app_align_y[:, 2] = safe_z

        p_grasp = top_p + np.array([0.0, 0.0, float(cfg.grasp_down_z)], dtype=np.float32)

        p_trp_lift_z = top_p.copy()
        p_trp_lift_z[:, 2] = safe_z

        p_trp_align_x = p_trp_lift_z.copy()
        p_trp_align_x[:, 0] = base_p[:, 0] - 0.015

        p_trp_align_y = base_p.copy()
        p_trp_align_y[:, 2] = safe_z
        p_trp_align_y[:, 0] = base_p[:, 0] - 0.015

        p_stack = base_p + np.array([0.0, 0.0, 2.0 * float(cfg.cube_half) + 0.005], dtype=np.float32)
        p_stack[:, 0] = base_p[:, 0] - 0.015

        p_retreat = base_p + np.array([0.0, 0.0, float(cfg.lift_dz) + 0.1], dtype=np.float32)
        p_home = np.tile(np.array([0.33502, 0.0, 0.11], dtype=np.float32), (B, 1))

        s = self.states
        tgt_pos = self.exec_pos.copy()
        grip_cmd = np.full((B,), float(cfg.gripper_open), dtype=np.float32)
        tgt_yaw = np.full((B,), np.nan, dtype=np.float32)

        def set_target(state_id: int, pos: np.ndarray, grip: float):
            mask = running & (s == state_id)
            if np.any(mask):
                tgt_pos[mask] = pos[mask]
                grip_cmd[mask] = float(grip)

        def set_target_yaw(state_id: int, pos: np.ndarray, grip: float, yaw_arr: np.ndarray):
            mask = running & (s == state_id)
            if np.any(mask):
                tgt_pos[mask] = pos[mask]
                grip_cmd[mask] = float(grip)
                tgt_yaw[mask] = yaw_arr[mask]

        set_target(self.ST_APP_LIFT_Z, p_app_lift_z, cfg.gripper_open)
        set_target(self.ST_APP_ALIGN_X, p_app_align_x, cfg.gripper_open)
        set_target(self.ST_APP_ALIGN_Y, p_app_align_y, cfg.gripper_open)
        set_target_yaw(self.ST_APP_ALIGN_YAW, p_app_align_y, cfg.gripper_open, self.latched_top_yaw + float(cfg.yaw_offset_grasp))
        set_target(self.ST_APP_DESCEND, p_grasp, cfg.gripper_open)

        set_target(self.ST_CLOSE, p_grasp, cfg.gripper_close)

        set_target(self.ST_TRP_LIFT_Z, p_trp_lift_z, cfg.gripper_close)
        set_target(self.ST_TRP_ALIGN_X, p_trp_align_x, cfg.gripper_close)
        set_target(self.ST_TRP_ALIGN_Y, p_trp_align_y, cfg.gripper_close)
        set_target_yaw(self.ST_TRP_ALIGN_YAW, p_trp_align_y, cfg.gripper_close, self.latched_base_yaw + float(cfg.yaw_offset_place))
        set_target(self.ST_TO_STACK, p_stack, cfg.gripper_close)

        set_target(self.ST_OPEN_HOLD, p_stack, cfg.gripper_open)
        set_target(self.ST_RETREAT, p_retreat, cfg.gripper_open)
        set_target_yaw(self.ST_TO_HOME, p_home, cfg.gripper_open, self.latched_start_yaw)

        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos, float(cfg.max_dp))

        want_rot = running & (~np.isnan(tgt_yaw)) & ((s == self.ST_APP_ALIGN_YAW) | (s == self.ST_TRP_ALIGN_YAW) | (s == self.ST_TO_HOME))

        desired_quat = self.exec_quat.copy()
        if np.any(want_rot):
            target_yaw = tgt_yaw[want_rot].astype(np.float32)
            start_yaw = self.latched_start_yaw[want_rot].astype(np.float32)
            delta_yaw = target_yaw - start_yaw

            symmetry = np.pi / 2.0
            delta_yaw = (delta_yaw + symmetry / 2.0) % symmetry - symmetry / 2.0

            r_delta = Rotation.from_euler("z", delta_yaw, degrees=False)
            r_start = Rotation.from_quat(self.latched_start_quat[want_rot])
            r_target = r_delta * r_start

            q = normalize_quat(r_target.as_quat().astype(np.float32))
            desired_quat[want_rot] = q
            self.exec_quat[want_rot] = q

        ref_pose_6d = self.env.robot.ref_ee_pose
        ref_pos = ref_pose_6d[:, :3]
        ref_euler = ref_pose_6d[:, 3:6]
        ref_quat = Rotation.from_euler("xyz", ref_euler, degrees=False).as_quat().astype(np.float32)

        rotvec_cmd = np.zeros((B, 3), dtype=np.float32)
        if np.any(want_rot):
            r_des = Rotation.from_quat(desired_quat[want_rot])
            r_ref = Rotation.from_quat(ref_quat[want_rot])
            r_err = r_des * r_ref.inv()
            rv = r_err.as_rotvec().astype(np.float32)

            mag = np.linalg.norm(rv, axis=1, keepdims=True) + 1e-9
            scale = np.minimum(1.0, float(cfg.max_dr) / mag)
            rv = rv * scale
            rotvec_cmd[want_rot] = rv * float(cfg.rot_gain)

        action = np.zeros((B, 7), dtype=np.float32)
        action[:, :3] = (self.exec_pos - ref_pos) * 0.5
        action[:, 3:6] = rotvec_cmd
        action[:, 6] = grip_cmd

        self._last_action[:] = action
        self.env.step(action)

        def is_reached(target_p: np.ndarray) -> np.ndarray:
            return np.linalg.norm(self.exec_pos - target_p, axis=1) < float(cfg.pos_tol)

        obs = self.env._state.obs
        ee_pose = obs.get("ee_pose", None)
        if ee_pose is not None and ee_pose.shape[1] == 7:
            ee_quat = ee_pose[:, 3:7].astype(np.float32)
        elif ee_pose is not None and ee_pose.shape[1] == 6:
            ee_quat = Rotation.from_euler("xyz", ee_pose[:, 3:6], degrees=False).as_quat().astype(np.float32)
        else:
            ee_quat = ref_quat.copy()

        curr_yaw = Rotation.from_quat(ee_quat).as_euler("xyz", degrees=False)[:, 2].astype(np.float32)

        def _check_reach_and_dwell(state_idx: int, target_p: np.ndarray) -> np.ndarray:
            in_state = running & (self.states == state_idx)
            reached = is_reached(target_p)
            just = in_state & reached & (self.state_reach_step == -1)
            if np.any(just):
                self.state_reach_step[just] = self.ctrl_step[just]
            dwell = (self.ctrl_step - self.state_reach_step) >= int(cfg.waypoint_dwell_steps)
            return in_state & (self.state_reach_step != -1) & dwell & reached

        def _check_yaw_and_dwell(state_idx: int, target_yaw_arr: np.ndarray) -> np.ndarray:
            in_state = running & (self.states == state_idx)
            dy = wrap_to_pi(curr_yaw - target_yaw_arr.astype(np.float32))
            symmetry = np.pi / 2.0
            dy = (dy + symmetry / 2.0) % symmetry - symmetry / 2.0
            ok = np.abs(dy) < float(cfg.yaw_tol)
            just = in_state & ok & (self.state_reach_step == -1)
            if np.any(just):
                self.state_reach_step[just] = self.ctrl_step[just]
            dwell = (self.ctrl_step - self.state_reach_step) >= int(cfg.waypoint_dwell_steps)
            return in_state & (self.state_reach_step != -1) & dwell & ok

        self._enter_state(_check_reach_and_dwell(self.ST_APP_LIFT_Z, p_app_lift_z), self.ST_APP_ALIGN_X)
        self._enter_state(_check_reach_and_dwell(self.ST_APP_ALIGN_X, p_app_align_x), self.ST_APP_ALIGN_Y)
        self._enter_state(_check_reach_and_dwell(self.ST_APP_ALIGN_Y, p_app_align_y), self.ST_APP_ALIGN_YAW)

        grasp_yaw_tgt = (self.latched_top_yaw + float(cfg.yaw_offset_grasp)).astype(np.float32)
        self._enter_state(_check_yaw_and_dwell(self.ST_APP_ALIGN_YAW, grasp_yaw_tgt), self.ST_APP_DESCEND)

        self._enter_state(_check_reach_and_dwell(self.ST_APP_DESCEND, p_grasp), self.ST_CLOSE)

        mask_close = running & (self.states == self.ST_CLOSE)
        if np.any(mask_close):
            done_close = (self.ctrl_step - self.state_enter_step) >= int(cfg.close_hold_steps)
            self._enter_state(mask_close & done_close, self.ST_TRP_LIFT_Z)

        self._enter_state(_check_reach_and_dwell(self.ST_TRP_LIFT_Z, p_trp_lift_z), self.ST_TRP_ALIGN_X)
        self._enter_state(_check_reach_and_dwell(self.ST_TRP_ALIGN_X, p_trp_align_x), self.ST_TRP_ALIGN_Y)
        self._enter_state(_check_reach_and_dwell(self.ST_TRP_ALIGN_Y, p_trp_align_y), self.ST_TRP_ALIGN_YAW)

        place_yaw_tgt = (self.latched_base_yaw + float(cfg.yaw_offset_place)).astype(np.float32)
        self._enter_state(_check_yaw_and_dwell(self.ST_TRP_ALIGN_YAW, place_yaw_tgt), self.ST_TO_STACK)

        self._enter_state(_check_reach_and_dwell(self.ST_TO_STACK, p_stack), self.ST_OPEN_HOLD)

        mask_open = running & (self.states == self.ST_OPEN_HOLD)
        if np.any(mask_open):
            is_stacked = self._check_stack_success(mask_open)
            self.stack_hold_counter[mask_open & is_stacked] += 1
            self.stack_hold_counter[mask_open & (~is_stacked)] = 0
            ready = self.stack_hold_counter >= int(cfg.stack_hold_steps)
            self._enter_state(mask_open & ready, self.ST_RETREAT)

        mask_ret = running & (self.states == self.ST_RETREAT)
        if np.any(mask_ret):
            self._enter_state(mask_ret & is_reached(p_retreat), self.ST_TO_HOME)

        mask_home = running & (self.states == self.ST_TO_HOME)
        if np.any(mask_home):
            self._enter_state(mask_home & is_reached(p_home), self.ST_DONE)

    # -----------------------------
    # Loop
    # -----------------------------
    def collect(self) -> None:
        cfg = self.cfg
        target_n = int(cfg.data_size)

        all_ids = np.arange(self.B, dtype=np.int64)
        self.start_episodes(all_ids, seed=int(cfg.seed))

        print(f"Starting StackColorBlocks Collection (dual-view, mediapy). Target: {target_n}")

        while self.saved_count < target_n:
            self._step_logic()

            running = self.active & (~self.done)

            sample_mask = running & ((self.ctrl_step % int(cfg.sample_every_steps)) == 0)
            for env_id in np.where(sample_mask)[0].tolist():
                self._capture_step(int(env_id))

            if cfg.save_video:
                render_mask = running & ((self.ctrl_step % int(cfg.render_every_steps)) == 0)
                for env_id in np.where(render_mask)[0].tolist():
                    self._write_video_frame(int(env_id))

            self.ctrl_step[running] += 1

            for i in range(self.B):
                if not running[i]:
                    continue
                fsm_done = (int(self.states[i]) == int(self.ST_DONE))
                timeout = (int(self.ctrl_step[i]) >= int(cfg.max_ctrl_steps))
                is_stacked = self._check_stack_success(np.eye(self.B, dtype=bool)[i])[i]
                if fsm_done or timeout:
                    self.done[i] = True
                    self.success[i] = bool(fsm_done and is_stacked)

            for i in range(self.B):
                if self.active[i] and self.done[i]:
                    self._finalize_episode(i)
                    if self.saved_count < target_n:
                        self.active[i] = False
                        new_seed = int(cfg.seed + self.attempted)
                        self.start_episodes(np.array([i], dtype=np.int64), seed=new_seed)
                    else:
                        self.active[i] = False

            now = time.perf_counter()
            if (now - self._last_log_t) > 2.0:
                print(f"[Collect] Saved: {self.saved_count}/{target_n} | Active: {int(self.active.sum())}")
                self._last_log_t = now

        print(f"Done. Saved to {cfg.save_dir}")
        self.close()

    def close(self) -> None:
        for env_id in range(self.B):
            self._close_writers_for_env(env_id)
        if hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--data_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no_video", action="store_true")

    # optional camera key overrides
    p.add_argument("--cam0", type=str, default=None)
    p.add_argument("--cam1", type=str, default=None)

    # jsonl fix knobs
    p.add_argument("--ee_z_offset", type=float, default=None)
    p.add_argument("--yaw_fix_bias", type=float, default=None)
    p.add_argument("--ee_quat_convention", type=str, default=None, choices=["xyzw", "wxyz"])

    args = p.parse_args()

    # DO NOT reflect default_factory from dataclass (was the crash reason).
    # Use explicit default here.
    default_cam_keys = ["pixels/view_0", "pixels/view_1"]
    cam_keys = default_cam_keys
    if args.cam0 is not None and args.cam1 is not None:
        cam_keys = [str(args.cam0), str(args.cam1)]
    elif args.cam0 is not None:
        cam_keys = [str(args.cam0)]

    cfg = CollectorCfg(
        save_dir=args.save_dir if args.save_dir is not None else CollectorCfg.save_dir,
        num_envs=args.num_envs if args.num_envs is not None else CollectorCfg.num_envs,
        data_size=args.data_size if args.data_size is not None else CollectorCfg.data_size,
        seed=args.seed if args.seed is not None else CollectorCfg.seed,
        save_video=(not args.no_video),
        cam_view_key=cam_keys,
        ee_z_offset=(args.ee_z_offset if args.ee_z_offset is not None else CollectorCfg.ee_z_offset),
        yaw_fix_bias=(args.yaw_fix_bias if args.yaw_fix_bias is not None else CollectorCfg.yaw_fix_bias),
        ee_quat_convention=(args.ee_quat_convention if args.ee_quat_convention is not None else CollectorCfg.ee_quat_convention),
    )

    runner = StackColorBlocksCollector(cfg)
    try:
        runner.collect()
    except KeyboardInterrupt:
        pass
    finally:
        runner.close()


if __name__ == "__main__":
    main()
