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
    data_size: int = 1
    num_envs: int = 1
    seed: int = 300
    save_dir: str = "./data/table30_hang_toothbrush_cup_env_collect_20" # Updated dir name

    # env control
    max_ctrl_steps: int = 1000 # 增加步数以容纳两段曼哈顿路径

    # motion
    max_dp: float = 0.005
    pos_tol: float = 0.001 # 提高精度

    # keypoints offsets (world frame offsets)
    grasp_offset: Tuple[float, float, float] = (0.0, -0.04, 0.02)
    # 这个高度将作为 Approach 阶段的安全平面高度
    pre_grasp_z: float = 0.05 

    pre_hang_offset: Tuple[float, float, float] = (-0.046, -0.15, 0.03)
    hang_offset: Tuple[float, float, float] = (-0.046, -0.02 , 0.03)
    retreat_dx: float = 0.10

    # gripper
    gripper_open: float = 0.0
    gripper_close: float = 0.75
    
    # timing / dwell
    close_hold_steps: int = 25
    release_hold_steps: int = 25
    waypoint_dwell_steps: int = 30  # 每个中间点停顿的帧数

    # sampling / render
    sample_every_steps: int = 1
    save_video: bool = True
    render_every_steps: int = 1
    video_fps: int = 30
    video_w: int = 640
    video_h: int = 480
    cam_view_key: Optional[str] = None

    # text fields
    subtask: Optional[str] = None
    prompt: Optional[str] = None


# -----------------------------------------------------------------------------
# Collector
# -----------------------------------------------------------------------------
class HangToothbrushCupCollector:
    # --- FSM States (Full Manhattan Path) ---
    
    # Phase 1: Approach (Grasp) - Manhattan [New]
    ST_APP_LIFT_Z = 0   # 初始位置垂直抬升到安全高度
    ST_APP_ALIGN_X = 1  # 移动 X 对齐杯子
    ST_APP_ALIGN_Y = 2  # 移动 Y 对齐杯子
    ST_APP_DESCEND = 3  # 下降到抓取点

    # Phase 2: Grasp
    ST_CLOSE = 4

    # Phase 3: Transport (Hang) - Manhattan [Renamed for consistency]
    ST_TRP_LIFT_Z = 5   # (Was ST_LIFT_ALIGN_Z) 垂直提起对齐挂钩高度
    ST_TRP_ALIGN_X = 6  # (Was ST_ALIGN_X) 平移 X 对齐挂钩
    ST_TRP_ALIGN_Y = 7  # (Was ST_GO_PRE_HANG) 平移 Y 到达 Pre-Hang
    ST_HANG_DOWN = 8

    # Phase 4: Release & End
    ST_RELEASE = 9
    ST_RETREAT = 10
    ST_GO_RESET = 11
    ST_DONE = 12

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
        
        # [Dwell Timer]
        self.state_reach_step = np.full(B, -1, dtype=np.int32)

        self.fixed_rpy = np.zeros((B, 3), dtype=np.float32)
        self.exec_pos = np.zeros((B, 3), dtype=np.float32)
        self.home_pos = np.zeros((B, 3), dtype=np.float32)
        self.hung_latched = np.zeros(B, dtype=bool)

        # Latch positions
        self.latched_start_pos = np.zeros((B, 3), dtype=np.float32) # [New]
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
        self.state_enter_step[env_ids] = 0
        
        # [Reset Dwell Timer]
        self.state_reach_step[env_ids] = -1

        obs = self.env._state.obs
        ee6_all = np.asarray(obs["ee_pose"], dtype=np.float32).reshape(self.B, -1)
        self.exec_pos[env_ids] = ee6_all[env_ids, :3]
        self.fixed_rpy[env_ids] = ee6_all[env_ids, 3:6]
        self.home_pos[env_ids] = ee6_all[env_ids, :3]
        
        # [New] 记录初始位置，用于规划 Approach 的起点
        self.latched_start_pos[env_ids] = ee6_all[env_ids, :3]

        data = self.env._state.data
        self.env.robot.reset_envs(data, done_mask)
        self.env.robot.update_reference(data)

        grasp_pose7 = np.asarray(self.grasp_site.get_pose(data), dtype=np.float32).reshape(self.B, -1)
        hook_pose7  = np.asarray(self.hook_site.get_pose(data), dtype=np.float32).reshape(self.B, -1)

        self.latched_grasp_pos[env_ids] = grasp_pose7[env_ids, :3]
        self.latched_hook_pos[env_ids]  = hook_pose7[env_ids, :3]
        
        # Start State
        self.states[env_ids] = self.ST_APP_LIFT_Z

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
        buf["is_success"].append(is_success) # Duplicate in original, keeping consistent
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
        # if self.success[env_id] and (self.saved_success < int(self.cfg.data_size)):
        if True :
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
        
        # [Reset reach timer]
        self.state_reach_step[mask] = -1

    def _step_logic(self) -> None:
        cfg = self.cfg
        B = self.B

        running = self.active & (~self.done)
        if not np.any(running):
            return
        
        start_p = self.latched_start_pos
        grasp_p = self.latched_grasp_pos + np.asarray(cfg.grasp_offset, dtype=np.float32).reshape(1, 3)
        hook_p = self.latched_hook_pos

        # --- 1. 定义关键点 (Full Manhattan Path) ---
        
        # 安全平面高度: 抓取位置 + pre_grasp_z
        safe_z = grasp_p[:, 2] + float(cfg.pre_grasp_z)
        
        # Phase 1: Approach Sequence
        # 1. Lift Z: 保持 Start 的 XY，提升 Z 到 safe_z
        p_app_lift_z = start_p.copy()
        p_app_lift_z[:, 2] = safe_z
        
        # 2. Align X: 移动 X 到 Grasp X，保持 Start Y，保持 safe_z
        p_app_align_x = p_app_lift_z.copy()
        p_app_align_x[:, 0] = grasp_p[:, 0]
        
        # 3. Align Y: 移动 Y 到 Grasp Y，此时 X 已对齐，Z 保持 safe_z
        # 此时正好位于 grasp_p 的正上方 (即原 pre_grasp_p)
        p_app_align_y = grasp_p.copy()
        p_app_align_y[:, 2] = safe_z
        
        # 4. Descend: 垂直下降到 Grasp Position
        p_app_descend = grasp_p
        
        # Phase 3: Transport Sequence
        # Targets for Hang
        pre_hang_p = hook_p + np.asarray(cfg.pre_hang_offset, dtype=np.float32).reshape(1, 3)
        hang_p = hook_p + np.asarray(cfg.hang_offset, dtype=np.float32).reshape(1, 3)

        # 5. Lift Z (Transport): 保持 Grasp 的 XY，只提升 Z 到 Pre-Hang 的高度
        # 这里用 pre_hang_p.z 作为运输层高度
        p_trp_lift_z = grasp_p.copy()
        p_trp_lift_z[:, 2] = pre_hang_p[:, 2] 

        # 6. Align X (Transport): 保持 Grasp 的 Y，只移动 X 到 Pre-Hang 的 X
        p_trp_align_x = p_trp_lift_z.copy()
        p_trp_align_x[:, 0] = pre_hang_p[:, 0]

        # 7. Align Y (Transport): 移动 Y 到 Pre-Hang 的 Y
        # 实际上这个点就是 pre_hang_p
        p_trp_align_y = pre_hang_p
        
        # Phase 4: End
        retreat_p = hang_p.copy()
        retreat_p[:, 0] -= float(cfg.retreat_dx)
        reset_p = self.home_pos

        # --- 2. 目标分配 ---
        tgt_pos = self.exec_pos.copy()
        grip_cmd = np.full((B,), float(cfg.gripper_open), dtype=np.float32)

        s = self.states
        
        # Helper to set target
        def set_target(state_id, pos, grip):
            mask = running & (s == state_id)
            if np.any(mask):
                tgt_pos[mask] = pos[mask]
                grip_cmd[mask] = grip

        # Phase 1: Approach
        set_target(self.ST_APP_LIFT_Z,  p_app_lift_z,  cfg.gripper_open)
        set_target(self.ST_APP_ALIGN_X, p_app_align_x, cfg.gripper_open)
        set_target(self.ST_APP_ALIGN_Y, p_app_align_y, cfg.gripper_open)
        set_target(self.ST_APP_DESCEND, p_app_descend, cfg.gripper_open)
        
        # Phase 2: Close
        set_target(self.ST_CLOSE,       p_app_descend, cfg.gripper_close)
        
        # Phase 3: Transport
        set_target(self.ST_TRP_LIFT_Z,  p_trp_lift_z,  cfg.gripper_close)
        set_target(self.ST_TRP_ALIGN_X, p_trp_align_x, cfg.gripper_close)
        set_target(self.ST_TRP_ALIGN_Y, p_trp_align_y, cfg.gripper_close)
        set_target(self.ST_HANG_DOWN,   hang_p,        cfg.gripper_close)
        
        # Phase 4: Release & End
        set_target(self.ST_RELEASE,     hang_p,        cfg.gripper_open)
        set_target(self.ST_RETREAT,     retreat_p,     cfg.gripper_open)
        set_target(self.ST_GO_RESET,    reset_p,       cfg.gripper_open)

        # Action Execution
        ref_pose_6d = self.env.robot.ref_ee_pose
        ref_pos = ref_pose_6d[:, :3]
        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos, float(cfg.max_dp))

        action = np.zeros((B, 7), dtype=np.float32)
        action[:, :3] = self.exec_pos - ref_pos
        action[:, :2] *= 0.5  # gain for XY
        action[:, 2] *= 0.5   # gain for Z
        action[:, 3:6] = 0 
        action[:, 6] = grip_cmd
        self._last_action[:] = action

        self.env.step(action)

        # Reach Check Helper
        def is_reached(p: np.ndarray) -> np.ndarray:
            return np.linalg.norm(self.exec_pos - p, axis=1) < float(cfg.pos_tol)
        
        # Dwell Check Helper
        def _check_reach_and_dwell(state_idx: int, target_p: np.ndarray) -> np.ndarray:
            in_state = running & (self.states == state_idx)
            reached = is_reached(target_p)
            
            # Record first reach time
            just_reached = in_state & reached & (self.state_reach_step == -1)
            if np.any(just_reached):
                self.state_reach_step[just_reached] = self.ctrl_step[just_reached]
            
            has_reached_before = (self.state_reach_step != -1)
            dwell_pass = (self.ctrl_step - self.state_reach_step) >= int(cfg.waypoint_dwell_steps)
            
            return in_state & has_reached_before & dwell_pass & reached

        # --- Phase 1: Approach Sequence ---
        # 1. Lift Z -> Align X
        done_app_lift = _check_reach_and_dwell(self.ST_APP_LIFT_Z, p_app_lift_z)
        self._enter_state(done_app_lift, self.ST_APP_ALIGN_X)
        
        # 2. Align X -> Align Y
        done_app_x = _check_reach_and_dwell(self.ST_APP_ALIGN_X, p_app_align_x)
        self._enter_state(done_app_x, self.ST_APP_ALIGN_Y)
        
        # 3. Align Y -> Descend
        done_app_y = _check_reach_and_dwell(self.ST_APP_ALIGN_Y, p_app_align_y)
        self._enter_state(done_app_y, self.ST_APP_DESCEND)
        
        # 4. Descend -> Close
        done_app_descend = _check_reach_and_dwell(self.ST_APP_DESCEND, p_app_descend)
        self._enter_state(done_app_descend, self.ST_CLOSE)
        
        # --- Phase 2: Close (Timer based) ---
        mask = running & (s == self.ST_CLOSE)
        if np.any(mask):
            time_in_state = self.ctrl_step - self.state_enter_step
            close_done = time_in_state >= int(cfg.close_hold_steps)
            self._enter_state(mask & close_done, self.ST_TRP_LIFT_Z)

        # --- Phase 3: Transport Sequence ---
        # 5. Lift Z -> Align X
        done_trp_lift = _check_reach_and_dwell(self.ST_TRP_LIFT_Z, p_trp_lift_z)
        self._enter_state(done_trp_lift, self.ST_TRP_ALIGN_X)
        
        # 6. Align X -> Align Y
        done_trp_x = _check_reach_and_dwell(self.ST_TRP_ALIGN_X, p_trp_align_x)
        self._enter_state(done_trp_x, self.ST_TRP_ALIGN_Y)
        
        # 7. Align Y -> Hang Down
        done_trp_y = _check_reach_and_dwell(self.ST_TRP_ALIGN_Y, p_trp_align_y)
        self._enter_state(done_trp_y, self.ST_HANG_DOWN)
        
        # 8. Hang Down -> Release
        done_hang = _check_reach_and_dwell(self.ST_HANG_DOWN, hang_p)
        self._enter_state(done_hang, self.ST_RELEASE)

        # --- Phase 4: Release & End ---
        # 9. Release -> Retreat (Timer based)
        in_rel = running & (self.states == self.ST_RELEASE)
        rel_done = in_rel & ((self.ctrl_step - self.state_enter_step) >= int(cfg.release_hold_steps))
        self._enter_state(rel_done, self.ST_RETREAT)

        # 10. Retreat -> Reset
        done_retreat = is_reached(retreat_p) # Simple reach is fine for retreat
        self._enter_state(running & (s == self.ST_RETREAT) & done_retreat, self.ST_GO_RESET)
        
        # 11. Reset -> Done
        done_reset = is_reached(reset_p)
        self._enter_state(running & (s == self.ST_GO_RESET) & done_reset, self.ST_DONE)

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