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
from gs_playground.src.manipulation.tasks.table30._02_stack_color_blocks_franka import (
    StackColorBlocksEnv,
    StackColorBlocksEnvCfg,
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def smooth_step_pos(curr: np.ndarray, tgt: np.ndarray, max_dp: float) -> np.ndarray:
    """
    平滑移动：限制单步最大位移 max_dp
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
    seed: int = 42
    save_dir: str = "./data/table30_stack_color_blocks_collect_full_manhattan"

    # env control
    max_ctrl_steps: int = 800

    # motion params
    max_dp: float = 0.005
    pos_tol: float = 0.001

    # rotation control (NEW)
    rot_gain: float = 0.6     # P gain for rotvec
    max_dr: float = 0.08      # max rotvec magnitude per step (rad)
    yaw_tol: float = 0.03     # yaw alignment tolerance (rad)

    # optional yaw offsets (NEW) - if you need gripper->cube frame offset
    yaw_offset_grasp: float = 0.0
    yaw_offset_place: float = 0.0

    # task specific offsets
    above_z: float = 0.00
    grasp_down_z: float = 0.0

    # 提升高度：作为所有水平移动的安全平面的相对高度
    lift_dz: float = 0.10
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
    video_w: int = 1280
    video_h: int = 720
    cam_view_key: Optional[str] = "pixels/view_0"

    # text fields
    subtask: Optional[str] = "Stack specific colored blocks."


# -----------------------------------------------------------------------------
# Collector
# -----------------------------------------------------------------------------
class StackColorBlocksCollector:
    # --- FSM States (Full Manhattan Path + Yaw Align) ---

    # Phase 1: Approach (Grasp) - Manhattan
    ST_APP_LIFT_Z = 0
    ST_APP_ALIGN_X = 1
    ST_APP_ALIGN_Y = 2
    ST_APP_ALIGN_YAW = 3   # [NEW] at top XY, rotate gripper to grasp yaw
    ST_APP_DESCEND = 4

    # Phase 2: Grasping
    ST_CLOSE = 5

    # Phase 3: Transport (Stack) - Manhattan
    ST_TRP_LIFT_Z = 6
    ST_TRP_ALIGN_X = 7
    ST_TRP_ALIGN_Y = 8
    ST_TRP_ALIGN_YAW = 9   # [NEW] at base XY, rotate gripper to place yaw
    ST_TO_STACK = 10

    # Phase 4: Release & Retreat
    ST_OPEN_HOLD = 11
    ST_RETREAT = 12
    ST_TO_HOME = 13
    ST_DONE = 14

    def __init__(self, cfg: CollectorCfg, env_cfg: Optional[StackColorBlocksEnvCfg] = None):
        self.cfg = cfg
        os.makedirs(cfg.save_dir, exist_ok=True)
        self.videos_dir = os.path.join(cfg.save_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

        # --- Env Setup ---
        self.env_cfg = env_cfg if env_cfg is not None else StackColorBlocksEnvCfg()
        self.env = StackColorBlocksEnv(self.env_cfg, num_envs=int(cfg.num_envs))
        self.env.reset()

        self.model = self.env.model
        self.B = int(cfg.num_envs)

        # Bodies
        self.cube_names = self.env_cfg.cube_names
        self.cube_bodies = self.env.cube_bodies

        self.cam_view_key = cfg.cam_view_key or "pixels/view_0"
        self.ep_subtask = np.array([cfg.subtask] * self.B, dtype=object)

        # --- Lifecycle ---
        self.active = np.zeros(self.B, dtype=bool)
        self.done = np.zeros(self.B, dtype=bool)
        self.success = np.zeros(self.B, dtype=bool)
        self.ctrl_step = np.zeros(self.B, dtype=np.int32)

        # --- FSM State ---
        self.states = np.zeros(self.B, dtype=np.int32)
        self.state_enter_step = np.zeros(self.B, dtype=np.int32)

        # Dwell timer (used for position states and yaw states)
        self.state_reach_step = np.full(self.B, -1, dtype=np.int32)

        self._attempt_id = np.zeros(self.B, dtype=np.int64)

        # Logic specific vars
        self.top_idx = np.zeros(self.B, dtype=np.int32)
        self.base_idx = np.zeros(self.B, dtype=np.int32)
        self.stack_hold_counter = np.zeros(self.B, dtype=np.int32)

        # Control Targets
        self.exec_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.exec_quat = np.zeros((self.B, 4), dtype=np.float32)  # xyzw

        # Latch positions
        self.latched_start_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.latched_top_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.latched_base_pos = np.zeros((self.B, 3), dtype=np.float32)

        # Latch orientation info (NEW)
        # [FIX] Added latched_start_quat and latched_start_yaw here
        self.latched_start_quat = np.zeros((self.B, 4), dtype=np.float32)
        self.latched_start_yaw = np.zeros((self.B,), dtype=np.float32) 
        
        self.latched_top_yaw = np.zeros((self.B,), dtype=np.float32)     # cube top yaw
        self.latched_base_yaw = np.zeros((self.B,), dtype=np.float32)    # cube base yaw

        # Buffers
        self.buffers: List[Dict[str, Any]] = [self._new_buffer() for _ in range(self.B)]
        self.video_writers: List[Optional[EpisodeVideoWriter]] = [None] * self.B
        self._tmp_video_paths: List[str] = [os.path.join(self.videos_dir, f"_tmp_env{i}.mp4") for i in range(self.B)]

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
        }

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

        for idx in env_ids:
            pose = all_poses[idx]
            self.exec_pos[idx] = pose[:3]
            self.latched_start_pos[idx] = pose[:3]

            # [FIX] Logic to properly get XY ZW quaternion from pose
            if len(pose) == 7:
                self.exec_quat[idx] = pose[3:]
            elif len(pose) == 6:
                euler = pose[3:]
                self.exec_quat[idx] = Rotation.from_euler("xyz", euler).as_quat()

            # [FIX] Latch start quaternion and start yaw
            self.latched_start_quat[idx] = self.exec_quat[idx].copy()
            # Extract absolute Yaw from start quaternion
            self.latched_start_yaw[idx] = Rotation.from_quat(self.exec_quat[idx]).as_euler("xyz")[2]

        self.top_idx[env_ids] = self.env.top_idx[env_ids]
        self.base_idx[env_ids] = self.env.base_idx[env_ids]

        cube_pose = np.stack(
            [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.cube_bodies],
            axis=1,
        )

        self.latched_top_pos[env_ids] = cube_pose[env_ids, self.top_idx[env_ids], :3]
        self.latched_base_pos[env_ids] = cube_pose[env_ids, self.base_idx[env_ids], :3]

        # Latch cube yaw for grasp/place
        # Assuming physics engine returns WXYZ, converting to XYZW for SciPy?
        # WARNING: Check if MotrixSim returns WXYZ or XYZW.
        # Standard Motrix/MuJoCo is usually WXYZ. SciPy is XYZW.
        # If your 'cube_pose' quat is [w, x, y, z], run this conversion:
        # q_xyzw = q_wxyz[[1,2,3,0]]
        # Here we assume it might be needed. Let's try standard SciPy access first.
        # If previous prints looked correct, maybe it's already XYZW or the sim is XYZW.
        # Safest bet is to check docs. Assuming XYZW for now based on your previous logs showing correct angles.
        
        top_q = cube_pose[env_ids, self.top_idx[env_ids], 3:7]   
        base_q = cube_pose[env_ids, self.base_idx[env_ids], 3:7] 
        
        top_yaw = Rotation.from_quat(top_q).as_euler("xyz", degrees=False)[:, 2]
        base_yaw = Rotation.from_quat(base_q).as_euler("xyz", degrees=False)[:, 2]
        
        self.latched_top_yaw[env_ids] = top_yaw.astype(np.float32) * -1
        self.latched_base_yaw[env_ids] = base_yaw.astype(np.float32) * -1

        # Start state
        self.states[env_ids] = self.ST_APP_LIFT_Z

        for env_id in env_ids.tolist():
            self.buffers[env_id] = self._new_buffer()
            if self.video_writers[env_id] is not None:
                self.video_writers[env_id].close()
                self.video_writers[env_id] = None
            if self.cfg.save_video:
                self._reset_video_writer(env_id)
            self._attempt_id[env_id] += 1
            self.attempted += 1

    def _reset_video_writer(self, env_id: int):
        tmp_path = self._tmp_video_paths[env_id]
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
        self.video_writers[env_id] = EpisodeVideoWriter(
            tmp_path, int(self.cfg.video_fps), (int(self.cfg.video_w), int(self.cfg.video_h))
        )

    def _capture_step(self, env_id: int) -> None:
        obs = self.env._state.obs
        buf = self.buffers[env_id]

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

    def _write_video_frame(self, env_id: int) -> None:
        vw = self.video_writers[env_id]
        if vw is None:
            return
        obs = self.env._state.obs
        if self.cam_view_key in obs:
            rgb = obs[self.cam_view_key][env_id]
            if rgb is not None:
                vw.write(rgb[..., ::-1].copy())
                self.buffers[env_id]["video_frames"] += 1

    def _finalize_episode(self, env_id: int) -> None:
        if self.video_writers[env_id]:
            self.video_writers[env_id].close()
            self.video_writers[env_id] = None

        if self.success[env_id]:
            if self.saved_count < self.cfg.data_size:
                ep_idx = int(self.saved_count)
                final_video_path = f"videos/episode_{ep_idx:05d}.mp4"
                abs_video_path = os.path.join(self.cfg.save_dir, final_video_path)

                if self.cfg.save_video and os.path.exists(self._tmp_video_paths[env_id]):
                    shutil.move(self._tmp_video_paths[env_id], abs_video_path)

                self._flush_jsonl(env_id, ep_idx, final_video_path)
                self.saved_count += 1
                print(f"[Success] Saved episode {ep_idx}. Total saved: {self.saved_count}")

        if os.path.exists(self._tmp_video_paths[env_id]):
            try:
                os.remove(self._tmp_video_paths[env_id])
            except:
                pass
        self.buffers[env_id] = self._new_buffer()

    def _flush_jsonl(self, env_id: int, ep_idx: int, vid_path: str):
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
                rec = {
                    "images_1": {"url": vid_path, "type": "video", "frame_idx": i},
                    "prompt": prompt,
                    "qpos": buf["qpos"][i],
                    "ee_pose": buf["ee_pose"][i],
                    "gripper": buf["gripper"][i],
                    "ctrl": buf["ctrl"][i],
                    "is_robot": True,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ----------------------------
    # Core Logic
    # ----------------------------
    def _enter_state(self, mask: np.ndarray, new_state: int) -> None:
        if not np.any(mask):
            return
        self.states[mask] = new_state
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

        # --- 1. 定义关键点 (Full Manhattan Path) ---
        safe_z = top_p[:, 2] + cfg.lift_dz

        # A. 抓取阶段 (Approach Phase)
        p_app_lift_z = start_p.copy()
        p_app_lift_z[:, 2] = safe_z

        p_app_align_x = p_app_lift_z.copy()
        p_app_align_x[:, 0] = top_p[:, 0]

        p_app_align_y = top_p.copy()
        p_app_align_y[:, 2] = safe_z

        p_grasp = top_p + np.array([0, 0, cfg.grasp_down_z], dtype=np.float32)

        # B. 搬运阶段 (Transport Phase)
        p_trp_lift_z = top_p.copy()
        p_trp_lift_z[:, 2] = safe_z

        p_trp_align_x = p_trp_lift_z.copy()
        p_trp_align_x[:, 0] = base_p[:, 0] - 0.015

        p_trp_align_y = base_p.copy()
        p_trp_align_y[:, 2] = safe_z
        p_trp_align_y[:, 0] = base_p[:, 0] - 0.015

        p_stack = base_p + np.array([0, 0, 2.0 * cfg.cube_half + 0.005], dtype=np.float32)
        p_stack[:, 0] = base_p[:, 0] - 0.015

        # C. 结束阶段
        p_retreat = base_p + np.array([0, 0, cfg.lift_dz + 0.05], dtype=np.float32)
        p_home = np.tile(np.array([0.4, 0.0, 0.5], dtype=np.float32), (B, 1))

        # --- 2. 目标分配 ---
        tgt_pos_curr = self.exec_pos.copy()
        grip_cmd = np.full((B,), cfg.gripper_open, dtype=np.float32)
        s = self.states

        # yaw target buffer: use NaN to mean "do not change target yaw"
        tgt_yaw = np.full((B,), np.nan, dtype=np.float32)

        def set_target(state_id: int, pos: np.ndarray, grip: float):
            mask = running & (s == state_id)
            if np.any(mask):
                tgt_pos_curr[mask] = pos[mask]
                grip_cmd[mask] = float(grip)

        def set_target_yaw(state_id: int, pos: np.ndarray, grip: float, yaw_arr: np.ndarray):
            mask = running & (s == state_id)
            if np.any(mask):
                tgt_pos_curr[mask] = pos[mask]
                grip_cmd[mask] = float(grip)
                tgt_yaw[mask] = yaw_arr[mask]

        # Phase 1: Approach
        set_target(self.ST_APP_LIFT_Z,  p_app_lift_z,  cfg.gripper_open)
        set_target(self.ST_APP_ALIGN_X, p_app_align_x, cfg.gripper_open)
        set_target(self.ST_APP_ALIGN_Y, p_app_align_y, cfg.gripper_open)

        # [NEW] Yaw align at top XY
        set_target_yaw(
            self.ST_APP_ALIGN_YAW,
            p_app_align_y,
            cfg.gripper_open,
            self.latched_top_yaw + float(cfg.yaw_offset_grasp),
        )

        set_target(self.ST_APP_DESCEND, p_grasp, cfg.gripper_open)

        # Phase 2: Close
        set_target(self.ST_CLOSE, p_grasp, cfg.gripper_close)

        # Phase 3: Transport
        set_target(self.ST_TRP_LIFT_Z,  p_trp_lift_z,  cfg.gripper_close)
        set_target(self.ST_TRP_ALIGN_X, p_trp_align_x, cfg.gripper_close)
        set_target(self.ST_TRP_ALIGN_Y, p_trp_align_y, cfg.gripper_close)

        # [NEW] Yaw align at base XY
        set_target_yaw(
            self.ST_TRP_ALIGN_YAW,
            p_trp_align_y,
            cfg.gripper_close,
            self.latched_base_yaw + float(cfg.yaw_offset_place),
        )

        set_target(self.ST_TO_STACK, p_stack, cfg.gripper_close)

        # Phase 4: Release & Home
        set_target(self.ST_OPEN_HOLD, p_stack, cfg.gripper_open)
        set_target(self.ST_RETREAT,   p_retreat, cfg.gripper_open)
        set_target(self.ST_TO_HOME,   p_home, cfg.gripper_open)

        # --- 3. 执行控制：位置 ---
        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos_curr, cfg.max_dp)

        # --- 3. 执行控制：姿态（yaw） ---
        
        # Determine "phase yaw" for each env
        grasp_phase = (
            (s >= self.ST_APP_ALIGN_YAW) &
            (s <= self.ST_TRP_ALIGN_Y)
        )
        place_phase = (s >= self.ST_TRP_ALIGN_YAW)

        phase_yaw = np.full((B,), np.nan, dtype=np.float32)
        phase_yaw[grasp_phase] = (self.latched_top_yaw[grasp_phase] + float(cfg.yaw_offset_grasp)).astype(np.float32) 
        phase_yaw[place_phase] = (self.latched_base_yaw[place_phase] + float(cfg.yaw_offset_place)).astype(np.float32) 

        # In yaw-align states, override with explicit tgt_yaw
        use_tgt = ~np.isnan(tgt_yaw)
        phase_yaw[use_tgt] = tgt_yaw[use_tgt].astype(np.float32)

        # Build desired quaternion (xyzw) using Rot Z synthesis
        want_rot = running & (~np.isnan(phase_yaw))
        desired_quat = self.exec_quat.copy() 

        if np.any(want_rot):
            # [FIX] Use Rotation Composition instead of Euler overwriting
            
            # 1. Targets & Starts
            target_yaw = phase_yaw[want_rot]
            start_yaw = self.latched_start_yaw[want_rot]
            
            # 2. Delta
            delta_yaw = target_yaw - start_yaw
            
            # 3. Symmetry Handling (90 degree symmetry for cubes)
            symmetry = np.pi / 2
            delta_yaw = (delta_yaw + symmetry/2) % symmetry - symmetry/2
            
            # 4. Compose: R_target = R_z(delta) * R_start
            # Construct Rotation objects
            r_delta = Rotation.from_euler('z', delta_yaw)
            r_start = Rotation.from_quat(self.latched_start_quat[want_rot])
            
            # Multiply (intrinsic vs extrinsic? Standard physics usually extrinsic global Z)
            # We want to rotate around GLOBAL Z axis.

            # If r_start is q_WB (Body in World), and we want q_WB_new = RotZ * q_WB
            r_target = r_delta * r_start
            desired_quat[want_rot] = r_target.as_quat().astype(np.float32)
            self.exec_quat[want_rot] = desired_quat[want_rot]

        # Reference pose from robot (pos + euler). We use it as baseline like your position control.
        ref_pose_6d = self.env.robot.ref_ee_pose  # (B,6): [x,y,z,rx,ry,rz] in euler
        ref_pos = ref_pose_6d[:, :3]
        ref_euler = ref_pose_6d[:, 3:6]
        ref_quat = Rotation.from_euler("xyz", ref_euler, degrees=False).as_quat().astype(np.float32)

        # Compute rotvec error relative to ref_quat
        rotvec_cmd = np.zeros((B, 3), dtype=np.float32)
        if np.any(want_rot):
            r_des = Rotation.from_quat(desired_quat[want_rot])
            r_ref = Rotation.from_quat(ref_quat[want_rot])
            r_err = r_des * r_ref.inv()
            rv = r_err.as_rotvec().astype(np.float32)  # (n,3)

            # clip rot magnitude per step
            mag = np.linalg.norm(rv, axis=1, keepdims=True) + 1e-9
            scale = np.minimum(1.0, float(cfg.max_dr) / mag)
            rv = rv * scale

            rotvec_cmd[want_rot] = rv * float(cfg.rot_gain)

        # --- action ---
        action = np.zeros((B, 7), dtype=np.float32)
        action[:, :3] = (self.exec_pos - ref_pos) * 0.5  # P gain
        action[:, 3:6] = rotvec_cmd
        action[:, 6] = grip_cmd

        self._last_action[:] = action
        self.env.step(action)

        # --- 4. 状态跳转 (with Dwell) ---
        def is_reached(target_p: np.ndarray) -> np.ndarray:
            return np.linalg.norm(self.exec_pos - target_p, axis=1) < float(cfg.pos_tol)

        # For yaw checks, use current ee_pose from obs (not ref)
        obs = self.env._state.obs
        ee_pose = obs.get("ee_pose", None)

        # fallback: if ee_pose is 6d (pos+euler), synthesize quat
        if ee_pose is not None and ee_pose.shape[1] == 7:
            ee_quat = ee_pose[:, 3:7].astype(np.float32)  # xyzw assumed
        elif ee_pose is not None and ee_pose.shape[1] == 6:
            ee_quat = Rotation.from_euler("xyz", ee_pose[:, 3:6], degrees=False).as_quat().astype(np.float32)
        else:
            # last resort: use ref_quat as current
            ee_quat = ref_quat.copy()

        curr_yaw = Rotation.from_quat(ee_quat).as_euler("xyz", degrees=False)[:, 2].astype(np.float32)

        # helper wrap (local)
        def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
            return (a + np.pi) % (2.0 * np.pi) - np.pi

        def _check_reach_and_dwell(state_idx: int, target_p: np.ndarray) -> np.ndarray:
            in_state = running & (self.states == state_idx)
            reached = is_reached(target_p)

            just_reached = in_state & reached & (self.state_reach_step == -1)
            if np.any(just_reached):
                self.state_reach_step[just_reached] = self.ctrl_step[just_reached]

            has_reached_before = (self.state_reach_step != -1)
            dwell_pass = (self.ctrl_step - self.state_reach_step) >= int(cfg.waypoint_dwell_steps)
            return in_state & has_reached_before & dwell_pass & reached

        def _check_yaw_and_dwell(state_idx: int, target_yaw: np.ndarray) -> np.ndarray:
            in_state = running & (self.states == state_idx)
            
            # [FIX] Yaw check needs to respect symmetry too, or use raw difference if symmetry already handled in command
            # Since command handles symmetry, we just check if robot reached the commanded yaw?
            # Or check if robot yaw is close to target yaw (modulo symmetry)?
            # Simplest is checking delta in 90-deg domain.
            
            dy = _wrap_to_pi(curr_yaw - target_yaw)
            # Modulo symmetry for check
            symmetry = np.pi / 2
            dy = (dy + symmetry/2) % symmetry - symmetry/2
            
            yaw_ok = np.abs(dy) < float(cfg.yaw_tol)

            just_reached = in_state & yaw_ok & (self.state_reach_step == -1)
            if np.any(just_reached):
                self.state_reach_step[just_reached] = self.ctrl_step[just_reached]

            has_reached_before = (self.state_reach_step != -1)
            dwell_pass = (self.ctrl_step - self.state_reach_step) >= int(cfg.waypoint_dwell_steps)
            return in_state & has_reached_before & dwell_pass & yaw_ok

        # --- Phase 1: Approach Sequence ---
        done_app_lift = _check_reach_and_dwell(self.ST_APP_LIFT_Z, p_app_lift_z)
        self._enter_state(done_app_lift, self.ST_APP_ALIGN_X)

        done_app_x = _check_reach_and_dwell(self.ST_APP_ALIGN_X, p_app_align_x)
        self._enter_state(done_app_x, self.ST_APP_ALIGN_Y)

        done_app_y = _check_reach_and_dwell(self.ST_APP_ALIGN_Y, p_app_align_y)
        self._enter_state(done_app_y, self.ST_APP_ALIGN_YAW)

        # [NEW] Align yaw -> Descend
        grasp_yaw_tgt = (self.latched_top_yaw + float(cfg.yaw_offset_grasp)).astype(np.float32)
        done_app_yaw = _check_yaw_and_dwell(self.ST_APP_ALIGN_YAW, grasp_yaw_tgt)
        self._enter_state(done_app_yaw, self.ST_APP_DESCEND)

        done_app_down = _check_reach_and_dwell(self.ST_APP_DESCEND, p_grasp)
        self._enter_state(done_app_down, self.ST_CLOSE)

        # --- Phase 2: Close ---
        mask = running & (s == self.ST_CLOSE)
        if np.any(mask):
            time_in_state = self.ctrl_step - self.state_enter_step
            closed_done = time_in_state >= int(cfg.close_hold_steps)
            self._enter_state(mask & closed_done, self.ST_TRP_LIFT_Z)

        # --- Phase 3: Transport Sequence ---
        done_trp_lift = _check_reach_and_dwell(self.ST_TRP_LIFT_Z, p_trp_lift_z)
        self._enter_state(done_trp_lift, self.ST_TRP_ALIGN_X)

        done_trp_x = _check_reach_and_dwell(self.ST_TRP_ALIGN_X, p_trp_align_x)
        self._enter_state(done_trp_x, self.ST_TRP_ALIGN_Y)

        done_trp_y = _check_reach_and_dwell(self.ST_TRP_ALIGN_Y, p_trp_align_y)
        self._enter_state(done_trp_y, self.ST_TRP_ALIGN_YAW)

        # [NEW] Align yaw at base -> Stack down
        place_yaw_tgt = (self.latched_base_yaw + float(cfg.yaw_offset_place)).astype(np.float32)
        done_trp_yaw = _check_yaw_and_dwell(self.ST_TRP_ALIGN_YAW, place_yaw_tgt)
        self._enter_state(done_trp_yaw, self.ST_TO_STACK)

        # --- Phase 4: Stack & End ---
        done_stack = _check_reach_and_dwell(self.ST_TO_STACK, p_stack)
        self._enter_state(done_stack, self.ST_OPEN_HOLD)

        mask = running & (s == self.ST_OPEN_HOLD)
        if np.any(mask):
            m_open = mask
            is_stacked = self._check_stack_success(m_open)
            self.stack_hold_counter[m_open & is_stacked] += 1
            self.stack_hold_counter[m_open & (~is_stacked)] = 0
            ready_retreat = (self.stack_hold_counter >= int(cfg.stack_hold_steps))
            self._enter_state(m_open & ready_retreat, self.ST_RETREAT)

        mask = running & (s == self.ST_RETREAT)
        if np.any(mask):
            self._enter_state(mask & is_reached(p_retreat), self.ST_TO_HOME)

        mask = running & (s == self.ST_TO_HOME)
        if np.any(mask):
            self._enter_state(mask & is_reached(p_home), self.ST_DONE)

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
        target_z = 2.0 * self.cfg.cube_half

        xy_ok = xy_dist < 0.03
        z_ok = np.abs(z_diff - target_z) < 0.02

        return (xy_ok & z_ok) & mask

    # ----------------------------
    # Main Loop
    # ----------------------------
    def collect(self) -> None:
        cfg = self.cfg
        target_n = cfg.data_size

        all_ids = np.arange(self.B, dtype=np.int64)
        self.start_episodes(all_ids, seed=cfg.seed)

        print(f"Starting Collection (Full Manhattan + Yaw Align). Target: {target_n}")

        while self.saved_count < target_n:
            self._step_logic()

            running = self.active & (~self.done)

            sample_mask = running & ((self.ctrl_step % int(cfg.sample_every_steps)) == 0)
            for env_id in np.where(sample_mask)[0]:
                self._capture_step(env_id)

            if cfg.save_video:
                render_mask = running & ((self.ctrl_step % int(cfg.render_every_steps)) == 0)
                for env_id in np.where(render_mask)[0]:
                    self._write_video_frame(env_id)

            self.ctrl_step[running] += 1

            for i in range(self.B):
                if not running[i]:
                    continue

                fsm_done = (self.states[i] == self.ST_DONE)
                timeout = (self.ctrl_step[i] >= int(cfg.max_ctrl_steps))
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
                        self.start_episodes(np.array([i]), seed=new_seed)
                    else:
                        self.active[i] = False

            now = time.perf_counter()
            if (now - self._last_log_t) > 2.0:
                print(f"[Collect] Saved: {self.saved_count}/{target_n} | Active: {self.active.sum()}")
                self._last_log_t = now

        print(f"Done. Saved to {cfg.save_dir}")
        self.close()

    def close(self):
        for vw in self.video_writers:
            if vw:
                vw.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--data_size", type=int, default=None)
    p.add_argument("--no_video", action="store_true")
    args = p.parse_args()

    cfg = CollectorCfg(
        save_dir=args.save_dir if args.save_dir else CollectorCfg.save_dir,
        num_envs=args.num_envs if args.num_envs else CollectorCfg.num_envs,
        data_size=args.data_size if args.data_size else CollectorCfg.data_size,
        save_video=(not args.no_video),
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