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
    # 避免除以零
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
    data_size: int = 1000
    num_envs: int = 50
    seed: int = 42
    save_dir: str = "./data/table30_stack_color_blocks_collect"

    # env control
    # [修改] 增加步数上限，给复位动作留出时间
    max_ctrl_steps: int = 500

    # motion params
    max_dp: float = 0.01 
    pos_tol: float = 0.005
    
    # task specific offsets (Task Logic Params)
    above_z: float = 0.00
    grasp_down_z: float = 0.00
    lift_dz: float = 0.05
    cube_half: float = 0.025
    
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
    subtask: Optional[str] = "Stack specific colored blocks."

# -----------------------------------------------------------------------------
# Collector
# -----------------------------------------------------------------------------
class StackColorBlocksCollector:
    # FSM States
    ST_TO_ABOVE_TOP = 0
    ST_TO_GRASP = 1
    ST_CLOSE = 2
    ST_LIFT = 3
    ST_TO_ABOVE_BASE = 4
    ST_TO_STACK = 5
    ST_OPEN_HOLD = 6
    ST_RETREAT = 7
    # [新增] 回家状态
    ST_TO_HOME = 8 
    ST_DONE = 9

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

        # Cam view key
        self.cam_view_key = cfg.cam_view_key or "pixels/view_0"

        # Metadata
        self.ep_subtask = np.array([cfg.subtask] * self.B, dtype=object)

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
        
        # Latch positions
        self.latched_top_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.latched_base_pos = np.zeros((self.B, 3), dtype=np.float32)

        # Buffers
        self.buffers: List[Dict[str, Any]] = [self._new_buffer() for _ in range(self.B)]
        self.video_writers: List[Optional[EpisodeVideoWriter]] = [None] * self.B
        self._tmp_video_paths: List[str] = [os.path.join(self.videos_dir, f"_tmp_env{i}.mp4") for i in range(self.B)]

        # Stats
        self.saved_count = 0
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
            "top_idx": [],
            "base_idx": [],
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
        self.state_enter_step[env_ids] = 0
        self.stack_hold_counter[env_ids] = 0

        # 3. Init Control Refs from Observation
        data = self.env._state.data
        self.env.robot.reset_envs(data, done_mask)
        self.env.robot.update_reference(data) 

        all_poses = self.env.robot.get_ee_pose(data)

        for idx in env_ids:
            pose = all_poses[idx]
            self.exec_pos[idx] = pose[:3]

            if len(pose) == 7:
                self.exec_quat[idx] = pose[3:]
            elif len(pose) == 6:
                euler = pose[3:]
                self.exec_quat[idx] = Rotation.from_euler('xyz', euler).as_quat()

        # 4. Logic: Sync with Env's hardcoded indices
        self.top_idx[env_ids] = self.env.top_idx[env_ids]
        self.base_idx[env_ids] = self.env.base_idx[env_ids]
        
        cube_pose = np.stack(
            [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.cube_bodies],
            axis=1,
        ) 
        
        self.latched_top_pos[env_ids] = cube_pose[env_ids, self.top_idx[env_ids], :3]
        self.latched_base_pos[env_ids] = cube_pose[env_ids, self.base_idx[env_ids], :3]
        
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
        
        buf["times"].append(float(self.ctrl_step[env_id] * 0.02)) 
        buf["logic_states"].append(int(self.states[env_id]))
        buf["qpos"].append(obs["qpos"][env_id].tolist())
        buf["ee_pose"].append(obs["ee_pose"][env_id].tolist())
        buf["gripper"].append(obs["gripper"][env_id].tolist())
        buf["ctrl"].append(self._last_action[env_id].tolist())
        
        info = self.env._state.info
        is_success = bool(self.success[env_id])
        buf["is_success"].append(is_success)
        buf["reward"].append(float(1.0 if is_success else 0.0))
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

        # [修改] 只有成功的 episode 才保存
        # 失败的 episode 会在 collect 循环中被丢弃并重置
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
        else:
            # 调试信息：失败则跳过
            pass
        
        if os.path.exists(self._tmp_video_paths[env_id]):
            try: os.remove(self._tmp_video_paths[env_id])
            except: pass
            
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
                    # "subtask": str(self.ep_subtask[env_id]),
                    "prompt": prompt,
                    "qpos": buf["qpos"][i],
                    "ee_pose": buf["ee_pose"][i],
                    "gripper": buf["gripper"][i],
                    "ctrl": buf["ctrl"][i],
                    # "reward": buf["reward"][i],
                    # "logic_state": buf["logic_states"][i],
                    # "time": buf["times"][i],
                    "is_robot": True,
                    # "success": bool(self.success[env_id]),
                    # "latched_top_pos": self.latched_top_pos[env_id].tolist() if i == 0 else [],
                    # "latched_base_pos": self.latched_base_pos[env_id].tolist() if i == 0 else [],
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

        top_p = self.latched_top_pos
        base_p = self.latched_base_pos
        
        # 1. 定义所有关键点
        p_above_top = top_p + np.array([0, 0, cfg.above_z])
        p_grasp     = top_p + np.array([0, 0, cfg.grasp_down_z])
        p_lift      = top_p + np.array([0, 0, cfg.lift_dz])
        p_above_base= base_p + np.array([0, 0, cfg.above_z + 0.05])
        p_stack     = base_p + np.array([0, 0, 2.0 * cfg.cube_half + 0.005])
        p_retreat   = base_p + np.array([0, 0, cfg.lift_dz])
        
        # [新增] Home Position: 桌面中央偏上 (0.4, 0.0, 0.5)
        p_home      = np.tile(np.array([0.4, 0.0, 0.5], dtype=np.float32), (B, 1))

        # 2. 根据当前状态确定 目标位置(tgt_pos_curr) 和 夹爪命令(grip_cmd)
        tgt_pos_curr = self.exec_pos.copy() # 默认不动
        grip_cmd = np.full((B,), cfg.gripper_open, dtype=np.float32)
        s = self.states

        # 使用掩码批量赋值
        mask_above = running & (s == self.ST_TO_ABOVE_TOP)
        if np.any(mask_above): tgt_pos_curr[mask_above] = p_above_top[mask_above]
            
        mask_grasp = running & (s == self.ST_TO_GRASP)
        if np.any(mask_grasp): tgt_pos_curr[mask_grasp] = p_grasp[mask_grasp]

        mask_close = running & (s == self.ST_CLOSE)
        if np.any(mask_close): 
            tgt_pos_curr[mask_close] = p_grasp[mask_close] # 闭合时维持在抓取点
            grip_cmd[mask_close] = cfg.gripper_close

        mask_lift = running & (s == self.ST_LIFT)
        if np.any(mask_lift):
            tgt_pos_curr[mask_lift] = p_lift[mask_lift]
            grip_cmd[mask_lift] = cfg.gripper_close

        mask_base = running & (s == self.ST_TO_ABOVE_BASE)
        if np.any(mask_base):
            tgt_pos_curr[mask_base] = p_above_base[mask_base]
            grip_cmd[mask_base] = cfg.gripper_close
            
        mask_stack = running & (s == self.ST_TO_STACK)
        if np.any(mask_stack):
            tgt_pos_curr[mask_stack] = p_stack[mask_stack]
            grip_cmd[mask_stack] = cfg.gripper_close
            
        mask_open = running & (s == self.ST_OPEN_HOLD)
        if np.any(mask_open):
            tgt_pos_curr[mask_open] = p_stack[mask_open] # 保持在堆叠点张开
            grip_cmd[mask_open] = cfg.gripper_open
            
        mask_retreat = running & (s == self.ST_RETREAT)
        if np.any(mask_retreat):
            tgt_pos_curr[mask_retreat] = p_retreat[mask_retreat]
            grip_cmd[mask_retreat] = cfg.gripper_open

        # [新增] Home 状态的目标设定
        mask_home = running & (s == self.ST_TO_HOME)
        if np.any(mask_home):
            tgt_pos_curr[mask_home] = p_home[mask_home]
            grip_cmd[mask_home] = cfg.gripper_open

        # 3. 更新虚拟轨迹 (exec_pos)
        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos_curr, cfg.max_dp)
        
        # 4. 下发控制 (Action)
        ref_pose_6d = self.env.robot.ref_ee_pose 
        ref_pos = ref_pose_6d[:, :3] # 真实位置


        # 映射状态 -> 目标

        action = np.zeros((B, 7), dtype=np.float32)
        action[:, :3] = (self.exec_pos - ref_pos) * 1.0 # P-Control
        action[:, 3:6] = 0 # 保持姿态
        action[:, 6] = grip_cmd
        
        self._last_action[:] = action
        self.env.step(action)

        # 5. 状态跳转逻辑
        def is_reached(target_p):
            return np.linalg.norm(self.exec_pos - target_p, axis=1) < cfg.pos_tol

        # ST_TO_ABOVE_TOP -> ST_TO_GRASP
        mask = running & (s == self.ST_TO_ABOVE_TOP)
        if np.any(mask):
            self._enter_state(mask & is_reached(p_above_top), self.ST_TO_GRASP)

        # ST_TO_GRASP -> ST_CLOSE
        mask = running & (s == self.ST_TO_GRASP)
        if np.any(mask):
            self._enter_state(mask & is_reached(p_grasp), self.ST_CLOSE)
        
        # ST_CLOSE -> ST_LIFT (时间判定)
        mask = running & (s == self.ST_CLOSE)
        if np.any(mask):
            time_in_state = self.ctrl_step - self.state_enter_step
            closed_done = time_in_state >= cfg.close_hold_steps
            self._enter_state(mask & closed_done, self.ST_LIFT)
        
        # ST_LIFT -> ST_TO_ABOVE_BASE
        mask = running & (s == self.ST_LIFT)
        if np.any(mask):
            self._enter_state(mask & is_reached(p_lift), self.ST_TO_ABOVE_BASE)
        
        # ST_TO_ABOVE_BASE -> ST_TO_STACK
        mask = running & (s == self.ST_TO_ABOVE_BASE)
        if np.any(mask):
            self._enter_state(mask & is_reached(p_above_base), self.ST_TO_STACK)

        # ST_TO_STACK -> ST_OPEN_HOLD
        mask = running & (s == self.ST_TO_STACK)
        if np.any(mask):
            self._enter_state(mask & is_reached(p_stack), self.ST_OPEN_HOLD)
        
        # ST_OPEN_HOLD -> ST_RETREAT
        mask = running & (s == self.ST_OPEN_HOLD)
        if np.any(mask):
            m_open = mask
            is_stacked = self._check_stack_success(m_open)
            self.stack_hold_counter[m_open & is_stacked] += 1
            self.stack_hold_counter[m_open & (~is_stacked)] = 0 
            
            ready_retreat = (self.stack_hold_counter >= cfg.stack_hold_steps)
            self._enter_state(m_open & ready_retreat, self.ST_RETREAT)
            
        # [修改] ST_RETREAT -> ST_TO_HOME (不是直接DONE)
        mask = running & (s == self.ST_RETREAT)
        if np.any(mask):
            self._enter_state(mask & is_reached(p_retreat), self.ST_TO_HOME)

        # [新增] ST_TO_HOME -> ST_DONE
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
        
        print(f"Starting Collection. Target (Total): {target_n}")
        
        while self.saved_count < target_n:
            self._step_logic()
            
            running = self.active & (~self.done)
            
            sample_mask = running & ((self.ctrl_step % cfg.sample_every_steps) == 0)
            for env_id in np.where(sample_mask)[0]:
                self._capture_step(env_id)
                
            if cfg.save_video:
                render_mask = running & ((self.ctrl_step % cfg.render_every_steps) == 0)
                for env_id in np.where(render_mask)[0]:
                    self._write_video_frame(env_id)
            
            self.ctrl_step[running] += 1
            
            for i in range(self.B):
                if not running[i]: continue
                
                fsm_done = (self.states[i] == self.ST_DONE)
                timeout = (self.ctrl_step[i] >= cfg.max_ctrl_steps)
                
                is_stacked = self._check_stack_success(np.eye(self.B, dtype=bool)[i])[i]
                
                if fsm_done or timeout:
                    self.done[i] = True
                    # [关键] 流程走完 + 堆叠成功 才算最终成功
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