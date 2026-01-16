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
    data_size: int = 50
    num_envs: int = 5
    seed: int = 42
    save_dir: str = "./data/table30_stack_color_blocks_collect"

    # env control
    max_ctrl_steps: int = 600

    # motion params (tuned for stacking)
    max_dp: float = 0.005
    pos_tol: float = 0.015
    
    # task specific offsets (Task Logic Params)
    above_z: float = 0.08
    grasp_down_z: float = 0.02
    lift_dz: float = 0.15
    cube_half: float = 0.0125 # XML size is 0.025 full size usually, but here logic implies radius
    
    # gripper
    gripper_open: float = 0.0
    gripper_close: float = 0.82
    close_hold_steps: int = 15
    stack_hold_steps: int = 10 

    # sampling / render
    sample_every_steps: int = 1
    save_video: bool = True
    render_every_steps: int = 1
    video_fps: int = 30
    video_w: int = 320
    video_h: int = 240
    cam_view_key: Optional[str] = "pixels/view_0"

    # text fields
    subtask: Optional[str] = "Stack one block on another."
    prompt: Optional[str] = "Stack the yellow block on the orange block." 


# -----------------------------------------------------------------------------
# Collector
# -----------------------------------------------------------------------------
class StackColorBlocksCollector:
    # FSM States matching the Runner logic
    ST_SAMPLE_PAIR = 0
    ST_TO_ABOVE_TOP = 1
    ST_TO_GRASP = 2
    ST_CLOSE = 3
    ST_LIFT = 4
    ST_TO_ABOVE_BASE = 5
    ST_TO_STACK = 6
    ST_OPEN_HOLD = 7
    ST_RETREAT = 8
    ST_DONE = 9

    def __init__(self, cfg: CollectorCfg, env_cfg: Optional[StackColorBlocksEnvCfg] = None):
        self.cfg = cfg
        os.makedirs(cfg.save_dir, exist_ok=True)
        self.videos_dir = os.path.join(cfg.save_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

        # --- Env Setup ---
        self.env_cfg = env_cfg if env_cfg is not None else StackColorBlocksEnvCfg()
        # Force gripper action mode if needed, usually 'eef' implies 7D control
        # self.env_cfg.action_mode = "eef" 

        self.env = StackColorBlocksEnv(self.env_cfg, num_envs=int(cfg.num_envs))
        self.env.reset()

        self.model = self.env.model
        self.B = int(cfg.num_envs)

        # Bodies for cubes (Blue, Yellow, Orange)
        # Assuming names are standardized in env cfg
        self.cube_names = self.env_cfg.cube_names
        self.cube_bodies = self.env.cube_bodies

        # cam view key
        self.cam_view_key = cfg.cam_view_key or "pixels/view_0"

        # Metadata
        self.ep_subtask = np.array([cfg.subtask] * self.B, dtype=object)
        self.ep_prompt = np.array([cfg.prompt] * self.B, dtype=object)

        # --- Lifecycle ---
        self.active = np.zeros(self.B, dtype=bool)
        self.done = np.zeros(self.B, dtype=bool)
        self.success = np.zeros(self.B, dtype=bool)
        self.ctrl_step = np.zeros(self.B, dtype=np.int32)
        self._attempt_id = np.zeros(self.B, dtype=np.int64)

        # --- FSM State ---
        self.states = np.zeros(self.B, dtype=np.int32)
        self.state_enter_step = np.zeros(self.B, dtype=np.int32)
        
        # Logic specific vars
        self.top_idx = np.zeros(self.B, dtype=np.int32)
        self.base_idx = np.zeros(self.B, dtype=np.int32)
        self.stack_hold_counter = np.zeros(self.B, dtype=np.int32)

        # Control Targets
        self.exec_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.exec_quat = np.zeros((self.B, 4), dtype=np.float32) # xyzw
        
        # Latch positions (Where the blocks were when we started)
        self.latched_top_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.latched_base_pos = np.zeros((self.B, 3), dtype=np.float32)

        # Buffers
        self.buffers: List[Dict[str, Any]] = [self._new_buffer() for _ in range(self.B)]
        self.video_writers: List[Optional[EpisodeVideoWriter]] = [None] * self.B
        self._tmp_video_paths: List[str] = [os.path.join(self.videos_dir, f"_tmp_env{i}.mp4") for i in range(self.B)]

        # Stats
        self.saved_success = 0
        self.attempted = 0
        self._last_log_t = time.perf_counter()

        # Cache last action for log
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
            # Metrics
            "top_idx": [],
            "base_idx": [],
            "dist_xy": [],
            "dist_z": [],
            "is_success": [],
            "video_frames": 0,
        }

    def start_episodes(self, env_ids: np.ndarray, seed: int) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if env_ids.size == 0:
            return

        # 1. Reset Env
        try:
            self.env._rng = np.random.default_rng(int(seed))
        except Exception:
            pass
        
        done_mask = np.zeros((self.B,), dtype=bool)
        done_mask[env_ids] = True
        self.env.reset(done=done_mask)

        # 2. Lifecycle
        self.active[env_ids] = True
        self.done[env_ids] = False
        self.success[env_ids] = False
        self.ctrl_step[env_ids] = 0
        self.states[env_ids] = self.ST_SAMPLE_PAIR 
        self.state_enter_step[env_ids] = 0
        self.stack_hold_counter[env_ids] = 0

        # 3. Init Control Refs from Observation
        data = self.env._state.data
        self.env.robot.reset_envs(data, done_mask)
        self.env.robot.update_reference(data) 

        # --- [修复开始] 兼容 6D (Euler) 和 7D (Quat) 返回值 ---
        # 获取所有环境的末端位姿
        all_poses = self.env.robot.get_ee_pose(data) # (B, 6) or (B, 7)

        for idx in env_ids:
            pose = all_poses[idx]
            self.exec_pos[idx] = pose[:3]

            if len(pose) == 7:
                # 已经是四元数 [x, y, z, qx, qy, qz, qw]
                self.exec_quat[idx] = pose[3:]
            elif len(pose) == 6:
                # 是欧拉角 [x, y, z, rx, ry, rz]，需要转为四元数
                # 假设通常 robot 接口返回的是 'xyz' 顺序的欧拉角
                euler = pose[3:]
                self.exec_quat[idx] = Rotation.from_euler('xyz', euler).as_quat()
            else:
                raise ValueError(f"Unexpected pose size: {len(pose)}")
        # --- [修复结束] ---

        # 4. Logic: Sample Pairs & Latch Positions
        rng = np.random.default_rng(seed)
        perms = np.argsort(rng.random((len(env_ids), 3)), axis=1)
        self.top_idx[env_ids] = perms[:, 0]
        self.base_idx[env_ids] = perms[:, 1]
        
        # Latch positions
        cube_pose = np.stack(
            [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.cube_bodies],
            axis=1,
        ) 
        
        sel_top = self.top_idx[env_ids]
        sel_base = self.base_idx[env_ids]
        
        self.latched_top_pos[env_ids] = cube_pose[env_ids, sel_top, :3]
        self.latched_base_pos[env_ids] = cube_pose[env_ids, sel_base, :3]
        
        # Transition immediately
        self.states[env_ids] = self.ST_TO_ABOVE_TOP

        # 5. Reset IO
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
            try: os.remove(tmp_path)
            except: pass
        self.video_writers[env_id] = EpisodeVideoWriter(
            tmp_path, int(self.cfg.video_fps), (int(self.cfg.video_w), int(self.cfg.video_h))
        )

    # ----------------------------
    # Capture / Write
    # ----------------------------
    def _capture_step(self, env_id: int) -> None:
        obs = self.env._state.obs
        buf = self.buffers[env_id]
        
        buf["times"].append(float(self.ctrl_step[env_id] * 0.02)) # Assume dt=0.02
        buf["logic_states"].append(int(self.states[env_id]))
        buf["qpos"].append(obs["qpos"][env_id].tolist())
        buf["ee_pose"].append(obs["ee_pose"][env_id].tolist())
        buf["gripper"].append(obs["gripper"][env_id].tolist())
        buf["ctrl"].append(self._last_action[env_id].tolist())
        
        # Reward / Success
        info = self.env._state.info
        is_success = False
        if "is_success" in info:
            is_success = bool(info["is_success"][env_id])
        elif "success" in info:
            is_success = bool(info["success"][env_id])
            
        buf["is_success"].append(is_success)
        buf["reward"].append(float(1.0 if is_success else 0.0))
        
        # Specific indices for this episode
        buf["top_idx"].append(int(self.top_idx[env_id]))
        buf["base_idx"].append(int(self.base_idx[env_id]))

    def _write_video_frame(self, env_id: int) -> None:
        vw = self.video_writers[env_id]
        if vw is None: return
        obs = self.env._state.obs
        if self.cam_view_key in obs:
            rgb = obs[self.cam_view_key][env_id]
            if rgb is not None:
                vw.write(rgb[..., ::-1].copy()) # RGB->BGR
                self.buffers[env_id]["video_frames"] += 1

    def _finalize_episode(self, env_id: int) -> None:
        if self.video_writers[env_id]:
            self.video_writers[env_id].close()
            self.video_writers[env_id] = None

        if self.success[env_id] and (self.saved_success < self.cfg.data_size):
            ep_idx = int(self.saved_success)
            final_video_path = f"videos/episode_{ep_idx:05d}.mp4"
            abs_video_path = os.path.join(self.cfg.save_dir, final_video_path)

            # Move video
            if self.cfg.save_video and os.path.exists(self._tmp_video_paths[env_id]):
                shutil.move(self._tmp_video_paths[env_id], abs_video_path)
            
            # Write JSONL
            self._flush_jsonl(env_id, ep_idx, final_video_path)
            self.saved_success += 1
        
        # Cleanup tmp
        if os.path.exists(self._tmp_video_paths[env_id]):
            try: os.remove(self._tmp_video_paths[env_id])
            except: pass
            
        self.buffers[env_id] = self._new_buffer()

    def _flush_jsonl(self, env_id: int, ep_idx: int, vid_path: str):
        path = os.path.join(self.cfg.save_dir, f"episode_{ep_idx:05d}.jsonl")
        buf = self.buffers[env_id]
        n = len(buf["times"])
        
        # Construct Prompt string dynamically if needed
        cube_names = ["Blue", "Yellow", "Orange"]
        t_name = cube_names[self.top_idx[env_id]]
        b_name = cube_names[self.base_idx[env_id]]
        prompt = f"Stack the {t_name.lower()} block on the {b_name.lower()} block."

        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                rec = {
                    "images_1": {"url": vid_path, "type": "video", "frame_idx": i},
                    "subtask": str(self.ep_subtask[env_id]),
                    "prompt": prompt,
                    "qpos": buf["qpos"][i],
                    "ee_pose": buf["ee_pose"][i],
                    "gripper": buf["gripper"][i],
                    "ctrl": buf["ctrl"][i],
                    "reward": buf["reward"][i],
                    "logic_state": buf["logic_states"][i],
                    "time": buf["times"][i],
                    "is_robot": True,
                    "success": bool(self.success[env_id]),
                    "latched_top_pos": self.latched_top_pos[env_id].tolist() if i == 0 else [],
                    "latched_base_pos": self.latched_base_pos[env_id].tolist() if i == 0 else [],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ----------------------------
    # Core Logic
    # ----------------------------
    def _enter_state(self, mask: np.ndarray, new_state: int) -> None:
        if not np.any(mask): return
        self.states[mask] = new_state
        self.state_enter_step[mask] = self.ctrl_step[mask].copy()

    def _step_logic(self) -> None:
        cfg = self.cfg
        B = self.B
        
        running = self.active & (~self.done)
        if not np.any(running): return

        # 1. Calculate Targets based on Latched Positions
        top_p = self.latched_top_pos
        base_p = self.latched_base_pos
        
        # Offsets
        # Shape (B, 3)
        above_top = top_p.copy(); above_top[:, 2] += cfg.above_z
        grasp_top = top_p.copy(); grasp_top[:, 2] += cfg.grasp_down_z
        lift_p    = above_top.copy(); lift_p[:, 2] += cfg.lift_dz
        above_base= base_p.copy(); above_base[:, 2] += (cfg.above_z + 0.05)
        # 2*half + buffer
        stack_p   = base_p.copy(); stack_p[:, 2] += (2.0 * cfg.cube_half + 0.005) 
        retreat_p = above_base.copy(); retreat_p[:, 2] += cfg.lift_dz

        tgt_pos = self.exec_pos.copy()
        grip_cmd = np.full((B,), cfg.gripper_open, dtype=np.float32)

        s = self.states
        
        # Mappings
        m_above_top = running & (s == self.ST_TO_ABOVE_TOP)
        m_grasp     = running & (s == self.ST_TO_GRASP)
        m_close     = running & (s == self.ST_CLOSE)
        m_lift      = running & (s == self.ST_LIFT)
        m_above_base= running & (s == self.ST_TO_ABOVE_BASE)
        m_stack     = running & (s == self.ST_TO_STACK)
        m_open      = running & (s == self.ST_OPEN_HOLD)
        m_retreat   = running & (s == self.ST_RETREAT)

        # Set Targets
        if np.any(m_above_top):
            tgt_pos[m_above_top] = above_top[m_above_top]
            grip_cmd[m_above_top] = cfg.gripper_open
        
        if np.any(m_grasp):
            tgt_pos[m_grasp] = grasp_top[m_grasp]
            grip_cmd[m_grasp] = cfg.gripper_open
            
        if np.any(m_close):
            tgt_pos[m_close] = grasp_top[m_close]
            grip_cmd[m_close] = cfg.gripper_close
            
        if np.any(m_lift):
            tgt_pos[m_lift] = lift_p[m_lift]
            grip_cmd[m_lift] = cfg.gripper_close
            
        if np.any(m_above_base):
            tgt_pos[m_above_base] = above_base[m_above_base]
            grip_cmd[m_above_base] = cfg.gripper_close
            
        if np.any(m_stack):
            tgt_pos[m_stack] = stack_p[m_stack]
            grip_cmd[m_stack] = cfg.gripper_close
            
        if np.any(m_open):
            tgt_pos[m_open] = stack_p[m_open]
            grip_cmd[m_open] = cfg.gripper_open
            
        if np.any(m_retreat):
            tgt_pos[m_retreat] = retreat_p[m_retreat]
            grip_cmd[m_retreat] = cfg.gripper_open

        # 2. Smooth Move
        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos, cfg.max_dp)
        
        # 3. Construct Action
        # Action is [dx, dy, dz, rx, ry, rz, grip] (Assuming P-Control on EE)
        # We need to convert maintained quat to euler for the action
        
        # Current Ref Pose from Robot (needed for delta pos)
        ref_pose_6d = self.env.robot.ref_ee_pose # (B, 6) usually
        ref_pos = ref_pose_6d[:, :3]
        
        # Rotation: Convert stored quat to euler 'xyz' matching standard robot controller expectations
        r_obj = Rotation.from_quat(self.exec_quat)
        euler_xyz = r_obj.as_euler('xyz', degrees=False).astype(np.float32)
        
        action = np.zeros((B, 7), dtype=np.float32)
        # Position Delta (P-Control)
        action[:, :3] = self.exec_pos - ref_pos 
        # Damping factor common in these collectors
        action[:, :3] *= 0.5 
        
        # Rotation: Here we usually send the DESIRED Euler angles if action mode is Absolute Rotation
        # OR delta if relative. 
        # Typically 'action_mode="eef"' in these frameworks takes [delta_pos, euler_target, grip] OR [delta_pos, delta_euler, grip]
        # Let's assume we send the TARGET Euler angles for rotation (holding steady).
        # If the robot drifts, this pulls it back to initial rotation.
        action[:, 3:6] = euler_xyz 
        
        action[:, 6] = grip_cmd
        
        self._last_action[:] = action
        self.env.step(action)

        # 4. State Transitions (Reach Checks)
        def _reach(targets):
            # Simple dist check
            d = np.linalg.norm(self.exec_pos - targets, axis=1)
            return d < cfg.pos_tol

        self._enter_state(m_above_top & _reach(above_top), self.ST_TO_GRASP)
        self._enter_state(m_grasp & _reach(grasp_top), self.ST_CLOSE)
        
        # Close Wait
        dt_close = self.ctrl_step - self.state_enter_step
        self._enter_state(m_close & (dt_close >= cfg.close_hold_steps), self.ST_LIFT)
        
        self._enter_state(m_lift & _reach(lift_p), self.ST_TO_ABOVE_BASE)
        self._enter_state(m_above_base & _reach(above_base), self.ST_TO_STACK)
        self._enter_state(m_stack & _reach(stack_p), self.ST_OPEN_HOLD)
        
        # Open/Hold Wait & Success Check logic
        # Here we just wait blindly, or check stack
        # Runner logic: increment hold counter if stack_success
        if np.any(m_open):
            is_stacked = self._check_stack_success(m_open)
            self.stack_hold_counter[m_open & is_stacked] += 1
            self.stack_hold_counter[m_open & (~is_stacked)] = 0 # reset if slip
            
            ready_retreat = (self.stack_hold_counter >= cfg.stack_hold_steps)
            self._enter_state(m_open & ready_retreat, self.ST_RETREAT)
            
        self._enter_state(m_retreat & _reach(retreat_p), self.ST_DONE)
        
        # Env Success Latch
        info = self.env._state.info
        if "is_success" in info:
            done_env = np.asarray(info["is_success"], dtype=bool)
            # If env says success, we can treat as done or wait for FSM.
            # Usually strict FSM is better for clean demos. 
            pass

    def _check_stack_success(self, mask: np.ndarray) -> np.ndarray:
        # Vectorized geometric check
        data = self.env._state.data
        cube_pose = np.stack([np.asarray(b.get_pose(data), dtype=np.float32) for b in self.cube_bodies], axis=1)
        
        # Get poses for current top/base per env
        row_ids = np.arange(self.B)
        t_idx = self.top_idx
        b_idx = self.base_idx
        
        tp = cube_pose[row_ids, t_idx, :3]
        bp = cube_pose[row_ids, b_idx, :3]
        
        xy_dist = np.linalg.norm(tp[:, :2] - bp[:, :2], axis=1)
        z_diff = tp[:, 2] - bp[:, 2]
        target_z = 2.0 * self.cfg.cube_half
        
        # Tolerances
        xy_ok = xy_dist < 0.02
        z_ok = np.abs(z_diff - target_z) < 0.015
        
        return (xy_ok & z_ok) & mask

    # ----------------------------
    # Main Loop
    # ----------------------------
    def collect(self) -> None:
        cfg = self.cfg
        target_n = cfg.data_size
        
        all_ids = np.arange(self.B, dtype=np.int64)
        self.start_episodes(all_ids, seed=cfg.seed)
        
        while self.saved_success < target_n:
            self._step_logic()
            
            running = self.active & (~self.done)
            
            # Sample
            sample_mask = running & ((self.ctrl_step % cfg.sample_every_steps) == 0)
            for env_id in np.where(sample_mask)[0]:
                self._capture_step(env_id)
                
            # Render
            if cfg.save_video:
                render_mask = running & ((self.ctrl_step % cfg.render_every_steps) == 0)
                for env_id in np.where(render_mask)[0]:
                    self._write_video_frame(env_id)
            
            self.ctrl_step[running] += 1
            
            # Check Termination
            for i in range(self.B):
                if not running[i]: continue
                
                # Check FSM done
                fsm_done = (self.states[i] == self.ST_DONE)
                # Check Timeout
                timeout = (self.ctrl_step[i] >= cfg.max_ctrl_steps)
                
                # Double check success at the end
                is_stacked = self._check_stack_success(np.array([True]))[0] if i == 0 else self._check_stack_success(np.eye(self.B, dtype=bool)[i])[i]
                
                if fsm_done or timeout:
                    self.done[i] = True
                    # Success if FSM finished AND geometry is valid
                    self.success[i] = bool(fsm_done and is_stacked)
            
            # Reset Done Envs
            for i in range(self.B):
                if self.active[i] and self.done[i]:
                    self._finalize_episode(i)
                    
                    if self.saved_success < target_n:
                        # Restart env
                        self.active[i] = False # Briefly mark inactive
                        # New seed based on total attempts
                        new_seed = int(cfg.seed + self.attempted)
                        self.start_episodes(np.array([i]), seed=new_seed)
                    else:
                        self.active[i] = False

            # Log
            now = time.perf_counter()
            if (now - self._last_log_t) > 2.0:
                print(f"[Collect] Saved: {self.saved_success}/{target_n} | Active: {self.active.sum()}")
                self._last_log_t = now
                
        print(f"Done. Saved to {cfg.save_dir}")
        self.close()

    def close(self):
        for vw in self.video_writers:
            if vw: vw.close()

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
        save_video=(not args.no_video)
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