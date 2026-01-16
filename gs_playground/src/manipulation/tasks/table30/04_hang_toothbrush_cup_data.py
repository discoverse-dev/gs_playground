from __future__ import annotations

import os
import json
import time
import shutil
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import motrixsim as mtx
from scipy.spatial.transform import Rotation
from gs_playground.src.manipulation.tasks.table30._04_hang_toothbrush_cup import (
    HangToothbrushCupEnv,
    HangToothbrushCupEnvCfg,
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def smooth_step_pos(curr: np.ndarray, tgt: np.ndarray, max_dp: float) -> np.ndarray:
    """
    curr/tgt: (B,3)
    """
    dp = tgt - curr
    n = np.linalg.norm(dp, axis=1, keepdims=True)
    s = np.minimum(1.0, float(max_dp) / (n + 1e-9))
    return curr + dp * s


# -----------------------------------------------------------------------------
# Video Writer
# -----------------------------------------------------------------------------
class EpisodeVideoWriter:
    def __init__(self, path: str, fps: int, size_wh: Tuple[int, int]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.size_wh = (int(size_wh[0]), int(size_wh[1]))
        self.vw = cv2.VideoWriter(path, fourcc, float(fps), self.size_wh)

    def write(self, bgr: Optional[np.ndarray]) -> None:
        if bgr is None:
            return
        if (bgr.shape[1], bgr.shape[0]) != self.size_wh:
            bgr = cv2.resize(bgr, self.size_wh, interpolation=cv2.INTER_AREA)
        self.vw.write(bgr)

    def close(self) -> None:
        if self.vw is not None and self.vw.isOpened():
            self.vw.release()
        self.vw = None


# -----------------------------------------------------------------------------
# Config (collector-only; env 已有的不要重复定义)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CollectorCfg:
    # dataset
    data_size: int = 1
    num_envs: int = 1
    seed: int = 0
    save_dir: str = "./data/table30_hang_toothbrush_cup_env_collect"

    # env control（只保留 collector 的“上限”）
    max_ctrl_steps: int = 5000

    # motion
    max_dp: float = 0.005
    pos_tol: float = 0.02

    # keypoints offsets (world frame offsets)
    grasp_offset: Tuple[float, float, float] = (0.0, 0.04, 0.0)
    pre_grasp_z: float = 0.05
    lift_height: float = 0.20

    pre_hang_offset: Tuple[float, float, float] = (-0.04, -0.07, 0.02)
    hang_offset: Tuple[float, float, float] = (0, 0.025, 0.0)
    retreat_dx: float = 0.10

    # gripper
    gripper_open: float = 0.0
    gripper_close: float = 0.82
    close_hold_steps: int = 25
    release_hold_steps: int = 25

    # sampling / render
    sample_every_steps: int = 1
    save_video: bool = True
    render_every_steps: int = 1
    video_fps: int = 30
    video_w: int = 800
    video_h: int = 600
    cam_view_key: Optional[str] = None  # None -> 默认 "pixels/view_0"

    # text fields（若 None，则默认继承 env cfg）
    subtask: Optional[str] = None   # None -> env._cfg.instruction
    prompt: Optional[str] = None    # None -> 从 instruction 简单派生


# -----------------------------------------------------------------------------
# Collector (success-only JSONL + MP4)
# -----------------------------------------------------------------------------
class HangToothbrushCupCollector:
    ST_GO_PRE_GRASP = 0
    ST_GO_GRASP = 1
    ST_CLOSE = 2
    ST_LIFT = 3
    ST_GO_PRE_HANG = 4
    ST_HANG_DOWN = 5
    ST_RELEASE = 6
    ST_RETREAT = 7
    ST_DONE = 8

    def __init__(self, cfg: CollectorCfg, env_cfg: Optional[HangToothbrushCupEnvCfg] = None):
        self.cfg = cfg
        os.makedirs(cfg.save_dir, exist_ok=True)
        self.videos_dir = os.path.join(cfg.save_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

        # --- env cfg：优先外部传入，否则用默认 ---
        self.env_cfg = env_cfg if env_cfg is not None else HangToothbrushCupEnvCfg()


        self.env = HangToothbrushCupEnv(self.env_cfg, num_envs=int(cfg.num_envs))
        self.env.reset()

        self.model = self.env.model
        self.B = int(cfg.num_envs)

        # cam view key
        self.cam_view_key = cfg.cam_view_key or "pixels/view_0"

        # episode prompt/subtask：默认继承 env
        instruction = str(getattr(self.env._cfg, "instruction", "") or "")
        self.ep_subtask = np.array([cfg.subtask or instruction] * self.B, dtype=object)
        self.ep_prompt = np.array([cfg.prompt or (instruction if instruction else "hang toothbrush cup")] * self.B, dtype=object)

        # --- per-env lifecycle ---
        B = self.B
        self.active = np.zeros(B, dtype=bool)
        self.done = np.zeros(B, dtype=bool)
        self.success = np.zeros(B, dtype=bool)
        self.ctrl_step = np.zeros(B, dtype=np.int32)
        self._attempt_id = np.zeros(B, dtype=np.int64)

        # --- FSM + targets ---
        self.states = np.zeros(B, dtype=np.int32)
        self.state_enter_step = np.zeros(B, dtype=np.int32)  # CLOSE/RELEASE dwell

        # fixed rpy per episode (来自 env obs)
        self.fixed_rpy = np.zeros((B, 3), dtype=np.float32)

        # exec target position (x,y,z) for EEF
        self.exec_pos = np.zeros((B, 3), dtype=np.float32)

        # latch sites per episode
        self.latched_grasp_pos = np.zeros((B, 3), dtype=np.float32)
        self.latched_hook_pos = np.zeros((B, 3), dtype=np.float32)

        # buffers for jsonl
        self.buffers: List[Dict[str, Any]] = [self._new_buffer() for _ in range(B)]

        # per-env video writer
        self.video_writers: List[Optional[EpisodeVideoWriter]] = [None] * B
        self._tmp_video_paths: List[str] = [os.path.join(self.videos_dir, f"_tmp_env{i}.mp4") for i in range(B)]

        # stats
        self.saved_success = 0
        self.attempted = 0
        self._last_log_t = time.perf_counter()

        # cached last action for ctrl logging
        self._last_action = np.zeros((B, 7), dtype=np.float32)

        # handles (sites exist on your env)
        self.grasp_site = self.env.grasp_site
        self.hook_site = self.env.hook_site



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
            # --- reuse env info ---
            "d_ee_obj": [],
            "d_obj_hook": [],
            "grasp_touch": [],
            "hook_touch": [],
            "is_grasped": [],
            "is_hung": [],
            "is_success": [],
            "success_now": [],
            "video_frames": 0,
        }

    def _next_seed(self, env_id: int) -> int:
        s = int(self.cfg.seed + 100000 * env_id + int(self._attempt_id[env_id]))
        self._attempt_id[env_id] += 1
        self.attempted += 1
        return s

    # ----------------------------
    # Small helpers: robust info reading
    # ----------------------------
    def _info_get_scalar(self, info: Dict[str, Any], env_id: int, keys: List[str], default: float = 0.0) -> float:
        for k in keys:
            if k in info and info[k] is not None:
                a = np.asarray(info[k]).reshape(-1)
                if a.size > env_id:
                    return float(a[env_id])
        return float(default)

    def _info_get_bool(self, info: Dict[str, Any], env_id: int, keys: List[str], default: bool = False) -> bool:
        for k in keys:
            if k in info and info[k] is not None:
                a = np.asarray(info[k]).reshape(-1)
                if a.size > env_id:
                    return bool(a[env_id])
        return bool(default)

    def start_episodes(self, env_ids: np.ndarray, seed: int) -> None:
        """
        Batch reset + batch init per-env episode state.
        Note: seed is batch-level (shared RNG stream), not per-env deterministic.
        """
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if env_ids.size == 0:
            return

        # 1) Set env RNG once for this batch reset
        try:
            self.env._rng = np.random.default_rng(int(seed))
        except Exception:
            pass

        # 2) Batch reset those envs
        done_mask = np.zeros((self.B,), dtype=bool)
        done_mask[env_ids] = True
        self.env.reset(done=done_mask)

        # 3) Lifecycle flags (vectorized)
        self.active[env_ids] = True
        self.done[env_ids] = False
        self.success[env_ids] = False
        self.ctrl_step[env_ids] = 0
        self.states[env_ids] = self.ST_GO_PRE_GRASP
        self.state_enter_step[env_ids] = 0

        # 4) Initialize exec_pos + fixed_rpy from obs (vectorized)
        obs = self.env._state.obs
        ee6_all = np.asarray(obs["ee_pose"], dtype=np.float32).reshape(self.B, -1)  # (B,6)
        self.exec_pos[env_ids] = ee6_all[env_ids, :3]
        self.fixed_rpy[env_ids] = ee6_all[env_ids, 3:6]

        # 5) Latch site positions (take full, index by env_ids)
        data = self.env._state.data
        self.env.robot.reset_envs(data, done_mask)
        self.env.robot.update_reference(data)

        grasp_pose7 = np.asarray(self.grasp_site.get_pose(data), dtype=np.float32).reshape(self.B, -1)
        hook_pose7  = np.asarray(self.hook_site.get_pose(data), dtype=np.float32).reshape(self.B, -1)

        self.latched_grasp_pos[env_ids] = grasp_pose7[env_ids, :3]
        self.latched_hook_pos[env_ids]  = hook_pose7[env_ids, :3]


        # 7) Per-env buffers/video/tmp path reset (I/O keep loop)
        for env_id in env_ids.tolist():
            self.buffers[env_id] = self._new_buffer()

            if self.video_writers[env_id] is not None:
                self.video_writers[env_id].close()
                self.video_writers[env_id] = None

            if self.cfg.save_video:
                tmp_path = self._tmp_video_paths[env_id]
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                self.video_writers[env_id] = EpisodeVideoWriter(
                    tmp_path, int(self.cfg.video_fps), (int(self.cfg.video_w), int(self.cfg.video_h))
                )

            # bookkeeping for attempts
            self._attempt_id[env_id] += 1
            self.attempted += 1

    # ----------------------------
    # Capture / write
    # ----------------------------
    def _capture_step(self, env_id: int) -> None:
        obs = self.env._state.obs
        info = self.env._state.info

        qpos = obs["qpos"][env_id].tolist()
        ee = obs["ee_pose"][env_id].tolist()
        grip = obs["gripper"][env_id].tolist()

        ctrl_vec = self._last_action[env_id].tolist()
        ctrl_dt = float(getattr(self.env._cfg, "ctrl_dt", 0.02))
        t_sec = float(self.ctrl_step[env_id] * ctrl_dt)

        rew = float(np.asarray(self.env._state.reward).reshape(-1)[env_id])

        # key candidates (兼容你不同 env 的命名)
        d_ee_obj = self._info_get_scalar(info, env_id, ["d_ee_cup", "d_ee_bottle", "d_ee_obj"], 0.0)
        d_obj_hook = self._info_get_scalar(info, env_id, ["d_cup_hook", "d_bottle_hook", "d_obj_hook"], 0.0)
        grasp_touch = self._info_get_scalar(info, env_id, ["grasp_touch", "bottle_grasp_touch"], 0.0)
        hook_touch = self._info_get_scalar(info, env_id, ["hook_touch", "rack_hook_touch"], 0.0)

        is_grasped = self._info_get_bool(info, env_id, ["is_grasped"], False)
        is_hung = self._info_get_bool(info, env_id, ["is_hung"], False)
        # env 可能写 is_success，也可能只写 success（TaskEnv 基类）
        is_success = self._info_get_bool(info, env_id, ["is_success", "success"], False)
        success_now = self._info_get_bool(info, env_id, ["success_now"], False)

        buf = self.buffers[env_id]
        buf["times"].append(t_sec)
        buf["logic_states"].append(int(self.states[env_id]))
        buf["qpos"].append(qpos)
        buf["ee_pose"].append(ee)
        buf["gripper"].append(grip)
        buf["ctrl"].append(ctrl_vec)
        buf["reward"].append(rew)

        buf["d_ee_obj"].append(d_ee_obj)
        buf["d_obj_hook"].append(d_obj_hook)
        buf["grasp_touch"].append(grasp_touch)
        buf["hook_touch"].append(hook_touch)
        buf["is_grasped"].append(is_grasped)
        buf["is_hung"].append(is_hung)
        buf["is_success"].append(is_success)
        buf["success_now"].append(success_now)

    def _write_video_frame(self, env_id: int) -> None:
        vw = self.video_writers[env_id]
        if vw is None:
            return
        obs = self.env._state.obs
        if self.cam_view_key not in obs:
            return
        rgb = obs[self.cam_view_key][env_id]
        if rgb is None:
            return
        bgr = rgb[..., ::-1].copy()
        vw.write(bgr)
        self.buffers[env_id]["video_frames"] += 1

    def _flush_episode_jsonl(self, env_id: int, ep_idx: int, video_rel_path: str) -> None:
        path = os.path.join(self.cfg.save_dir, f"episode_{ep_idx:05d}.jsonl")
        buf = self.buffers[env_id]
        n = len(buf["times"])

        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                legacy_state = buf["qpos"][i] + buf["gripper"][i]
                rec = {
                    "images_1": {"url": video_rel_path, "type": "video", "frame_idx": i},
                    "subtask": str(self.ep_subtask[env_id]),
                    "prompt": str(self.ep_prompt[env_id]),
                    "state": legacy_state,
                    "qpos": buf["qpos"][i],
                    "ee_pose": buf["ee_pose"][i],
                    "ctrl": buf["ctrl"][i],
                    "gripper": buf["gripper"][i],
                    "reward": float(buf["reward"][i]),
                    "is_robot": True,
                    "logic_state": int(buf["logic_states"][i]),
                    "time": float(buf["times"][i]),
                    "success": bool(self.success[env_id]),
                    # metrics
                    "d_ee_obj": float(buf["d_ee_obj"][i]),
                    "d_obj_hook": float(buf["d_obj_hook"][i]),
                    "grasp_touch": float(buf["grasp_touch"][i]),
                    "hook_touch": float(buf["hook_touch"][i]),
                    "is_grasped": bool(buf["is_grasped"][i]),
                    "is_hung": bool(buf["is_hung"][i]),
                    "is_success_env": bool(buf["is_success"][i]),
                    "success_now": bool(buf["success_now"][i]),
                }

                # 只在第一帧写一次，避免每帧重复
                if i == 0:
                    rec["latched_grasp_pos"] = self.latched_grasp_pos[env_id].tolist()
                    rec["latched_hook_pos"] = self.latched_hook_pos[env_id].tolist()

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _finalize_episode(self, env_id: int) -> None:
        if self.video_writers[env_id] is not None:
            self.video_writers[env_id].close()
            self.video_writers[env_id] = None

        tmp_path = self._tmp_video_paths[env_id]

        if self.success[env_id] and (self.saved_success < int(self.cfg.data_size)):
            ep_idx = int(self.saved_success)
            final_video_abs = os.path.join(self.videos_dir, f"episode_{ep_idx:05d}.mp4")
            video_rel_path = f"videos/episode_{ep_idx:05d}.mp4"

            if self.cfg.save_video:
                try:
                    if os.path.exists(final_video_abs):
                        os.remove(final_video_abs)
                except Exception:
                    pass
                try:
                    os.replace(tmp_path, final_video_abs)
                except Exception:
                    try:
                        shutil.copy2(tmp_path, final_video_abs)
                        os.remove(tmp_path)
                    except Exception:
                        pass

            self._flush_episode_jsonl(env_id, ep_idx, video_rel_path)
            self.saved_success += 1
        else:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

        self.buffers[env_id] = self._new_buffer()

    # ----------------------------
    # Core logic (vectorized)
    # ----------------------------
    def _enter_state(self, mask: np.ndarray, new_state: int) -> None:
        if not np.any(mask):
            return
        self.states[mask] = int(new_state)
        self.state_enter_step[mask] = self.ctrl_step[mask].copy()

    def _step_logic(self) -> None:
        cfg = self.cfg
        B = self.B

        running = self.active & (~self.done)
        if not np.any(running):
            return

        grasp_p = self.latched_grasp_pos + np.asarray(cfg.grasp_offset, dtype=np.float32).reshape(1, 3)
        pre_grasp_p = grasp_p.copy()
        pre_grasp_p[:, 2] += float(cfg.pre_grasp_z)

        lift_p = grasp_p.copy()
        lift_p[:, 2] += float(cfg.lift_height)

        hook_p = self.latched_hook_pos
        pre_hang_p = hook_p + np.asarray(cfg.pre_hang_offset, dtype=np.float32).reshape(1, 3)
        hang_p = hook_p + np.asarray(cfg.hang_offset, dtype=np.float32).reshape(1, 3)

        retreat_p = hang_p.copy()
        retreat_p[:, 0] -= float(cfg.retreat_dx)

        tgt_pos = self.exec_pos.copy()
        grip_cmd = np.full((B,), float(cfg.gripper_open), dtype=np.float32)

        s = self.states
        print(s)
        m0 = running & (s == self.ST_GO_PRE_GRASP)
        m1 = running & (s == self.ST_GO_GRASP)
        m2 = running & (s == self.ST_CLOSE)
        m3 = running & (s == self.ST_LIFT)
        m4 = running & (s == self.ST_GO_PRE_HANG)
        m5 = running & (s == self.ST_HANG_DOWN)
        m6 = running & (s == self.ST_RELEASE)
        m7 = running & (s == self.ST_RETREAT)

        if np.any(m0):
            tgt_pos[m0] = pre_grasp_p[m0]
            grip_cmd[m0] = float(cfg.gripper_open)

        if np.any(m1):
            tgt_pos[m1] = grasp_p[m1]
            grip_cmd[m1] = float(cfg.gripper_open)

        if np.any(m2):
            tgt_pos[m2] = grasp_p[m2]
            grip_cmd[m2] = float(cfg.gripper_close)

        if np.any(m3):
            tgt_pos[m3] = lift_p[m3]
            grip_cmd[m3] = float(cfg.gripper_close)

        if np.any(m4):
            tgt_pos[m4] = pre_hang_p[m4]
            grip_cmd[m4] = float(cfg.gripper_close)

        if np.any(m5):
            tgt_pos[m5] = hang_p[m5]
            grip_cmd[m5] = float(cfg.gripper_close)

        if np.any(m6):
            tgt_pos[m6] = hang_p[m6]
            grip_cmd[m6] = float(cfg.gripper_open)

        if np.any(m7):
            tgt_pos[m7] = retreat_p[m7]
            grip_cmd[m7] = float(cfg.gripper_open)

        ref_pose_6d = self.env.robot.ref_ee_pose   # (B,6)


        ref_pos = ref_pose_6d[:, :3]
        ref_rpy = ref_pose_6d[:, 3:6]

        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos, float(cfg.max_dp))

        action = np.zeros((B, 7), dtype=np.float32)
        action[:, :3] = self.exec_pos - ref_pos
        action[:, :2] *= 0.7
        action[:, 2] *= 0.7
        action[:, 3:6] = 0 
        action[:, 6] = grip_cmd
        self._last_action[:] = action

        self.env.step(action)

        def _reach(p: np.ndarray) -> np.ndarray:
            return np.linalg.norm(self.exec_pos - p, axis=1) < float(cfg.pos_tol)

        reach_pre_grasp = _reach(pre_grasp_p)
        reach_grasp = _reach(grasp_p)
        reach_lift = _reach(lift_p)
        reach_pre_hang = _reach(pre_hang_p)
        reach_hang = _reach(hang_p)
        reach_retreat = _reach(retreat_p)

        self._enter_state(running & (s == self.ST_GO_PRE_GRASP) & reach_pre_grasp, self.ST_GO_GRASP)
        self._enter_state(running & (s == self.ST_GO_GRASP) & reach_grasp, self.ST_CLOSE)

        in_close = running & (self.states == self.ST_CLOSE)
        close_done = in_close & ((self.ctrl_step - self.state_enter_step) >= int(cfg.close_hold_steps))
        self._enter_state(close_done, self.ST_LIFT)

        self._enter_state(running & (self.states == self.ST_LIFT) & reach_lift, self.ST_GO_PRE_HANG)
        self._enter_state(running & (self.states == self.ST_GO_PRE_HANG) & reach_pre_hang, self.ST_HANG_DOWN)
        self._enter_state(running & (self.states == self.ST_HANG_DOWN) & reach_hang, self.ST_RELEASE)

        in_rel = running & (self.states == self.ST_RELEASE)
        rel_done = in_rel & ((self.ctrl_step - self.state_enter_step) >= int(cfg.release_hold_steps))
        self._enter_state(rel_done, self.ST_RETREAT)

        self._enter_state(running & (self.states == self.ST_RETREAT) & reach_retreat, self.ST_DONE)

        # done by env success latch（兼容 is_success / success）
        info = self.env._state.info
        is_success = np.asarray(
            info.get("is_success", info.get("success", np.zeros((B,), dtype=np.bool_))),
            dtype=np.bool_,
        ).reshape(-1)
        done_by_env = running & is_success
        if np.any(done_by_env):
            self.states[done_by_env] = self.ST_DONE

    # ----------------------------
    # Collect loop
    # ----------------------------
    def collect(self) -> None:
        cfg = self.cfg
        target = int(cfg.data_size)

        all_ids = np.arange(self.B, dtype=np.int64)
        self.start_episodes(all_ids, seed=int(cfg.seed))

        while self.saved_success < target:
            self._step_logic()

            running = self.active & (~self.done)

            sample_mask = running & ((self.ctrl_step % int(cfg.sample_every_steps)) == 0)
            for env_id in np.where(sample_mask)[0].tolist():
                self._capture_step(env_id)

            if cfg.save_video:
                render_mask = running & ((self.ctrl_step % int(cfg.render_every_steps)) == 0)
                for env_id in np.where(render_mask)[0].tolist():
                    self._write_video_frame(env_id)

            self.ctrl_step[running] += 1

            info = self.env._state.info
            is_success = np.asarray(
                info.get("is_success", info.get("success", np.zeros((self.B,), dtype=np.bool_))),
                dtype=np.bool_,
            ).reshape(-1)

            for i in range(self.B):
                if (not self.active[i]) or self.done[i]:
                    continue
                timeout = int(self.ctrl_step[i]) >= int(cfg.max_ctrl_steps)
                finished = bool(is_success[i]) or (int(self.states[i]) == self.ST_DONE)
                if finished or timeout:
                    self.done[i] = True
                    self.success[i] = bool(is_success[i])

            for i in range(self.B):
                if (not self.active[i]) or (not self.done[i]):
                    continue

                self._finalize_episode(i)

                if self.saved_success >= target:
                    self.active[i] = False
                    continue

                restart_ids = np.where(self.active & self.done)[0]
                if restart_ids.size > 0:
                    # pick a batch seed; simplest: advance by attempted count (or time-based)
                    batch_seed = int(cfg.seed + self.attempted)
                    # clear done flags BEFORE start_episodes
                    self.done[restart_ids] = False
                    self.start_episodes(restart_ids, seed=batch_seed)

            now = time.perf_counter()
            if (now - self._last_log_t) >= 2.0:
                print(
                    f"[collect] active={int(np.sum(self.active))}/{self.B} "
                    f"saved_success={int(self.saved_success)}/{target} attempted={int(self.attempted)}"
                )
                self._last_log_t = now

        print(f"[DONE] saved_success={self.saved_success}/{target}, attempted={self.attempted}")
        print(f"Saved to: {cfg.save_dir}")

    def close(self) -> None:
        for vw in self.video_writers:
            if vw is not None:
                vw.close()

    def get_ee_pose(self, data: mtx.SceneData) -> np.ndarray:
        # Return 6D pose: XYZ + RPY (Roll, Pitch, Yaw)
        # motrixsim EE pose is [x, y, z, qx, qy, qz, qw] (xyzw)
        pose_7d = np.asarray(self.ee_site.get_pose(data), dtype=np.float32)
        pos = pose_7d[..., :3]
        quat_xyzw = pose_7d[..., 3:]
        
        # scipy expects xyzw, so we can use directly
        r = Rotation.from_quat(quat_xyzw)
        # zyx -> [yaw, pitch, roll]
        euler_zyx = r.as_euler('zyx', degrees=False)
        # flip to [roll, pitch, yaw]
        rpy = np.flip(euler_zyx, axis=-1)
        
        return np.concatenate([pos, rpy], axis=-1).astype(np.float32)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--data_size", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--max_ctrl_steps", type=int, default=None)

    # env cfg override（只允许改 env 已有字段，不在 collector 里重复定义）
    p.add_argument("--action_mode", type=str, default=None, choices=["eef", "eef_relative", "joint"])
    args = p.parse_args()

    # env cfg：如需覆盖 action_mode，在这里改（collector 不再持有 action_mode）
    env_cfg = HangToothbrushCupEnvCfg()
    if args.action_mode is not None:
        env_cfg.action_mode = str(args.action_mode)

    cfg = CollectorCfg(
        save_dir=args.save_dir if args.save_dir is not None else CollectorCfg.save_dir,
        data_size=args.data_size if args.data_size is not None else CollectorCfg.data_size,
        num_envs=args.num_envs if args.num_envs is not None else CollectorCfg.num_envs,
        seed=args.seed if args.seed is not None else CollectorCfg.seed,
        save_video=(not args.no_video),
        max_ctrl_steps=args.max_ctrl_steps if args.max_ctrl_steps is not None else CollectorCfg.max_ctrl_steps,
    )

    runner = HangToothbrushCupCollector(cfg, env_cfg=env_cfg)
    try:
        runner.collect()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
