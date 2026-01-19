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
# Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CollectorCfg:
    # dataset
    data_size: int = 5
    num_envs: int = 5
    seed: int = 300
    save_dir: str = "./data/table30_hang_toothbrush_cup_env_collect_refined"

    # env control
    max_ctrl_steps: int = 500 # 稍微增加一点，因为中间停顿多了

    # motion
    max_dp: float = 0.005
    pos_tol: float = 0.001

    # keypoints offsets (world frame offsets)
    grasp_offset: Tuple[float, float, float] = (0.0, -0.04, 0.02)
    pre_grasp_z: float = 0.05
    # lift_height: float = 0.20 # Deprecated in favor of pre-hang alignment logic

    pre_hang_offset: Tuple[float, float, float] = (-0.047, -0.15, 0.03)
    hang_offset: Tuple[float, float, float] = (-0.047, -0.02 , 0.03)
    retreat_dx: float = 0.10

    # gripper
    gripper_open: float = 0.0
    gripper_close: float = 0.82
    
    # timing / dwell
    close_hold_steps: int = 25
    release_hold_steps: int = 25
    waypoint_dwell_steps: int = 30  # NEW: 每个中间点停顿的帧数

    # sampling / render
    sample_every_steps: int = 1
    save_video: bool = True
    render_every_steps: int = 1
    video_fps: int = 30
    video_w: int = 320
    video_h: int = 240
    cam_view_key: Optional[str] = None

    # text fields
    subtask: Optional[str] = None
    prompt: Optional[str] = None


# -----------------------------------------------------------------------------
# Collector
# -----------------------------------------------------------------------------
class HangToothbrushCupCollector:
    # 状态机定义更新
    ST_GO_PRE_GRASP = 0
    ST_GO_GRASP = 1
    ST_CLOSE = 2
    ST_LIFT_ALIGN_Z = 3   # NEW: 提升至与 Pre-Hang 等高
    ST_ALIGN_X = 4        # NEW: 平移 X 轴对齐
    ST_GO_PRE_HANG = 5    # NEW: 平移 Y 轴到达 Pre-Hang (X,Z已对齐)
    ST_HANG_DOWN = 6
    ST_RELEASE = 7
    ST_RETREAT = 8
    ST_GO_RESET = 9
    ST_DONE = 10

    def __init__(self, cfg: CollectorCfg, env_cfg: Optional[HangToothbrushCupEnvCfg] = None):
        self.cfg = cfg
        os.makedirs(cfg.save_dir, exist_ok=True)
        self.videos_dir = os.path.join(cfg.save_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

        self.env_cfg = env_cfg if env_cfg is not None else HangToothbrushCupEnvCfg()
        self.env = HangToothbrushCupEnv(self.env_cfg, num_envs=int(cfg.num_envs))
        self.env.reset()

        self.model = self.env.model
        self.B = int(cfg.num_envs)
        self.cam_view_key = cfg.cam_view_key or "pixels/view_0"

        instruction = str(getattr(self.env._cfg, "instruction", "") or "")
        self.ep_subtask = np.array([cfg.subtask or instruction] * self.B, dtype=object)
        self.ep_prompt = np.array([cfg.prompt or (instruction if instruction else "hang toothbrush cup")] * self.B, dtype=object)

        B = self.B
        self.active = np.zeros(B, dtype=bool)
        self.done = np.zeros(B, dtype=bool)
        self.success = np.zeros(B, dtype=bool)
        self.ctrl_step = np.zeros(B, dtype=np.int32)
        self._attempt_id = np.zeros(B, dtype=np.int64)

        self.states = np.zeros(B, dtype=np.int32)
        self.state_enter_step = np.zeros(B, dtype=np.int32)

        self.fixed_rpy = np.zeros((B, 3), dtype=np.float32)
        self.exec_pos = np.zeros((B, 3), dtype=np.float32)
        self.home_pos = np.zeros((B, 3), dtype=np.float32)
        self.hung_latched = np.zeros(B, dtype=bool)

        self.latched_grasp_pos = np.zeros((B, 3), dtype=np.float32)
        self.latched_hook_pos = np.zeros((B, 3), dtype=np.float32)

        self.buffers: List[Dict[str, Any]] = [self._new_buffer() for _ in range(B)]
        self.video_writers: List[Optional[EpisodeVideoWriter]] = [None] * B
        self._tmp_video_paths: List[str] = [os.path.join(self.videos_dir, f"_tmp_env{i}.mp4") for i in range(B)]

        self.saved_success = 0
        self.attempted = 0
        self._last_log_t = time.perf_counter()
        self._last_action = np.zeros((B, 7), dtype=np.float32)

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
        self.states[env_ids] = self.ST_GO_PRE_GRASP
        self.state_enter_step[env_ids] = 0

        obs = self.env._state.obs
        ee6_all = np.asarray(obs["ee_pose"], dtype=np.float32).reshape(self.B, -1)
        self.exec_pos[env_ids] = ee6_all[env_ids, :3]
        self.fixed_rpy[env_ids] = ee6_all[env_ids, 3:6]
        self.home_pos[env_ids] = ee6_all[env_ids, :3]

        data = self.env._state.data
        self.env.robot.reset_envs(data, done_mask)
        self.env.robot.update_reference(data)

        grasp_pose7 = np.asarray(self.grasp_site.get_pose(data), dtype=np.float32).reshape(self.B, -1)
        hook_pose7  = np.asarray(self.hook_site.get_pose(data), dtype=np.float32).reshape(self.B, -1)

        self.latched_grasp_pos[env_ids] = grasp_pose7[env_ids, :3]
        self.latched_hook_pos[env_ids]  = hook_pose7[env_ids, :3]

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
            self._attempt_id[env_id] += 1
            self.attempted += 1

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

        d_ee_obj = self._info_get_scalar(info, env_id, ["d_ee_cup", "d_ee_obj"], 0.0)
        d_obj_hook = self._info_get_scalar(info, env_id, ["d_cup_hook", "d_obj_hook"], 0.0)
        grasp_touch = self._info_get_scalar(info, env_id, ["grasp_touch"], 0.0)
        hook_touch = self._info_get_scalar(info, env_id, ["hook_touch"], 0.0)

        is_grasped = self._info_get_bool(info, env_id, ["is_grasped"], False)
        is_hung = self._info_get_bool(info, env_id, ["is_hung"], False)
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
                    "prompt": str(self.ep_prompt[env_id]),
                    "state": legacy_state,
                    "qpos": buf["qpos"][i],
                    "ee_pose": buf["ee_pose"][i],
                    "ctrl": buf["ctrl"][i],
                    "gripper": buf["gripper"][i],
                }
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
        # if True :
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

        # --- Keypoints Calculation ---
        # 1. Grasp
        grasp_p = self.latched_grasp_pos + np.asarray(cfg.grasp_offset, dtype=np.float32).reshape(1, 3)
        pre_grasp_p = grasp_p.copy()
        pre_grasp_p[:, 2] += float(cfg.pre_grasp_z)

        # 2. Pre-Hang & Hang
        hook_p = self.latched_hook_pos
        pre_hang_p = hook_p + np.asarray(cfg.pre_hang_offset, dtype=np.float32).reshape(1, 3)
        hang_p = hook_p + np.asarray(cfg.hang_offset, dtype=np.float32).reshape(1, 3)

        # 3. Intermediate Waypoints (Manhattan Path)
        # ST_LIFT_ALIGN_Z: 保持 Grasp 的 X,Y，只提升 Z 到 Pre-Hang 的高度
        lift_z_tgt = grasp_p.copy()
        lift_z_tgt[:, 2] = pre_hang_p[:, 2] 

        # ST_ALIGN_X: 保持 Grasp 的 Y，保持 Pre-Hang 的 Z，只移动 X 到 Pre-Hang 的 X
        align_x_tgt = lift_z_tgt.copy()
        align_x_tgt[:, 0] = pre_hang_p[:, 0]

        # ST_GO_PRE_HANG: 移动 Y 到 Pre-Hang 的 Y (X, Z 已经对齐)
        # 实际上这个点就是 pre_hang_p

        retreat_p = hang_p.copy()
        retreat_p[:, 0] -= float(cfg.retreat_dx)
        reset_p = self.home_pos

        tgt_pos = self.exec_pos.copy()
        grip_cmd = np.full((B,), float(cfg.gripper_open), dtype=np.float32)

        s = self.states
        
        # State Masks
        m0 = running & (s == self.ST_GO_PRE_GRASP)
        m1 = running & (s == self.ST_GO_GRASP)
        m2 = running & (s == self.ST_CLOSE)
        m3 = running & (s == self.ST_LIFT_ALIGN_Z)  
        m4 = running & (s == self.ST_ALIGN_X)       
        m5 = running & (s == self.ST_GO_PRE_HANG)   
        m6 = running & (s == self.ST_HANG_DOWN)
        m7 = running & (s == self.ST_RELEASE)
        m8 = running & (s == self.ST_RETREAT)
        m9 = running & (s == self.ST_GO_RESET)

        # Target Assignment
        if np.any(m0):
            tgt_pos[m0] = pre_grasp_p[m0]
            grip_cmd[m0] = float(cfg.gripper_open)
        if np.any(m1):
            tgt_pos[m1] = grasp_p[m1]
            grip_cmd[m1] = float(cfg.gripper_open)
        if np.any(m2):
            tgt_pos[m2] = grasp_p[m2]
            grip_cmd[m2] = float(cfg.gripper_close)
        
        # --- New Sequence Targets ---
        if np.any(m3): # LIFT Z
            tgt_pos[m3] = lift_z_tgt[m3]
            grip_cmd[m3] = float(cfg.gripper_close)
        if np.any(m4): # ALIGN X
            tgt_pos[m4] = align_x_tgt[m4]
            grip_cmd[m4] = float(cfg.gripper_close)
        if np.any(m5): # ALIGN Y (GO PRE HANG)
            tgt_pos[m5] = pre_hang_p[m5]
            grip_cmd[m5] = float(cfg.gripper_close)
        # ----------------------------

        if np.any(m6):
            tgt_pos[m6] = hang_p[m6]
            grip_cmd[m6] = float(cfg.gripper_close)
        if np.any(m7):
            tgt_pos[m7] = hang_p[m7]
            grip_cmd[m7] = float(cfg.gripper_open)
        if np.any(m8):
            tgt_pos[m8] = retreat_p[m8]
            grip_cmd[m8] = float(cfg.gripper_open)
        if np.any(m9):
            tgt_pos[m9] = reset_p[m9]
            grip_cmd[m9] = float(cfg.gripper_open)

        # Action Execution
        ref_pose_6d = self.env.robot.ref_ee_pose
        ref_pos = ref_pose_6d[:, :3]
        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos, float(cfg.max_dp))


        action = np.zeros((B, 7), dtype=np.float32)
        action[:, :3] = self.exec_pos - ref_pos
        action[:, :2] *= 0.5 # gain for XY
        action[:, 2] *= 0.5    # gain for Z
        action[:, 3:6] = 0 
        action[:, 6] = grip_cmd
        self._last_action[:] = action

        self.env.step(action)

        # Reach Check Helper
        def _reach(p: np.ndarray) -> np.ndarray:
            return np.linalg.norm(self.exec_pos - p, axis=1) < float(cfg.pos_tol)
        
        # Dwell Check Helper
        def _dwell_done(state_idx: int) -> np.ndarray:
            # Check if we have stayed in this state long enough
            in_state = running & (self.states == state_idx)
            steps_passed = self.ctrl_step - self.state_enter_step
            return in_state & (steps_passed >= int(cfg.waypoint_dwell_steps))

        reach_pre_grasp = _reach(pre_grasp_p)
        reach_grasp = _reach(grasp_p)
        
        reach_lift_z = _reach(lift_z_tgt)
        reach_align_x = _reach(align_x_tgt)
        reach_pre_hang = _reach(pre_hang_p)
        
        reach_hang = _reach(hang_p)
        reach_retreat = _reach(retreat_p)
        reach_reset = _reach(reset_p)

        # Transitions
        # 1. Grasp Sequence
        self._enter_state(running & (s == self.ST_GO_PRE_GRASP) & reach_pre_grasp, self.ST_GO_GRASP)
        self._enter_state(running & (s == self.ST_GO_GRASP) & reach_grasp, self.ST_CLOSE)
        
        # Close dwell
        in_close = running & (self.states == self.ST_CLOSE)
        close_done = in_close & ((self.ctrl_step - self.state_enter_step) >= int(cfg.close_hold_steps))
        self._enter_state(close_done, self.ST_LIFT_ALIGN_Z) # Jump to Lift Z

        # 2. Manhattan Move Sequence (with dwells)
        # Lift Z -> dwell -> Align X
        self._enter_state(running & (s == self.ST_LIFT_ALIGN_Z) & reach_lift_z & _dwell_done(self.ST_LIFT_ALIGN_Z), self.ST_ALIGN_X)
        
        # Align X -> dwell -> Align Y (Pre Hang)
        self._enter_state(running & (s == self.ST_ALIGN_X) & reach_align_x & _dwell_done(self.ST_ALIGN_X), self.ST_GO_PRE_HANG)
        
        # Pre Hang -> dwell -> Hang Down
        self._enter_state(running & (s == self.ST_GO_PRE_HANG) & reach_pre_hang & _dwell_done(self.ST_GO_PRE_HANG), self.ST_HANG_DOWN)

        # 3. Hang & Release Sequence
        # Hang Down -> dwell (reuse same logic or immediate? Using immediate for now, usually hang needs contact)
        # If you want hang dwell, add _dwell_done(self.ST_HANG_DOWN)
        self._enter_state(running & (s == self.ST_HANG_DOWN) & reach_hang, self.ST_RELEASE)

        in_rel = running & (self.states == self.ST_RELEASE)
        rel_done = in_rel & ((self.ctrl_step - self.state_enter_step) >= int(cfg.release_hold_steps))
        self._enter_state(rel_done, self.ST_RETREAT)

        self._enter_state(running & (self.states == self.ST_RETREAT) & reach_retreat, self.ST_GO_RESET)
        self._enter_state(running & (self.states == self.ST_GO_RESET) & reach_reset, self.ST_DONE)

        # Success Check
        info = self.env._state.info
        is_success = np.asarray(
            info.get("is_success", info.get("success", np.zeros((B,), dtype=np.bool_))),
            dtype=np.bool_,
        ).reshape(-1)
        self.hung_latched = self.hung_latched | (running & is_success)
        done_by_env = running & is_success
        if np.any(done_by_env):
            self.states[done_by_env] = self.ST_DONE

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

              # 6) mark done/success (per-env)
            for i in range(self.B):
                if (not self.active[i]) or self.done[i]:
                    continue

                timeout = int(self.ctrl_step[i]) >= int(cfg.max_ctrl_steps)
                finished = bool(is_success[i]) or (int(self.states[i]) == int(self.ST_DONE))

                if finished or timeout:
                    self.done[i] = True
                    self.success[i] = bool(is_success[i])


            # 7) finalize ALL done envs first (no restart inside per-env loop!)
            done_ids = np.where(self.active & self.done)[0]
            for env_id in done_ids.tolist():
                self._finalize_episode(int(env_id))

            # 8) if reached target, stop remaining envs and exit loop naturally
            if self.saved_success >= target:
                self.active[done_ids] = False
            else:
                # 9) restart done envs in one batch (only those still active)
                restart_ids = done_ids[self.active[done_ids]]
                if restart_ids.size > 0:
                    batch_seed = int(cfg.seed + self.attempted)
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
    p.add_argument("--action_mode", type=str, default=None)
    args = p.parse_args()

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