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
from scipy.spatial.transform import Rotation

# 引入你定义好的 Env
from gs_playground.src.manipulation.tasks.table30._06_put_cup_on_coaster import (
    CupOnCoasterEnv,
    CupOnCoasterEnvCfg,
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
    data_size: int = 1
    num_envs: int = 1
    seed: int = 1234
    save_dir: str = "./data/table30_cup_on_coaster_collect"

    # env control
    max_ctrl_steps: int = 600

    # motion params
    max_dp: float = 0.005
    pos_tol: float = 0.01
    
    # task specific offsets
    above_z: float = 0.15       # 悬停高度
    grasp_down_z: float = 0.00  # 抓取高度偏移 (相对于物体中心)
    lift_dz: float = 0.10       # 抬起高度
    place_z_offset: float = 0.05 # 放置时相对于杯垫的高度 (防止砸穿)
    
    # gripper
    gripper_open: float = 0.0
    gripper_close: float = 0.85
    close_hold_steps: int = 20  # 闭合后等待步数(让物理稳定)
    place_hold_steps: int = 10  # 放置后等待步数

    # sampling / render
    sample_every_steps: int = 1
    save_video: bool = True
    render_every_steps: int = 2
    video_fps: int = 30
    video_w: int = 320
    video_h: int = 240
    cam_view_key: Optional[str] = "pixels/view_0"

    # text fields
    subtask: str = "Place the cup onto the coaster."
    prompt: str = "Place the cup onto the coaster."

# -----------------------------------------------------------------------------
# Collector
# -----------------------------------------------------------------------------
class CupOnCoasterCollector:
    # FSM States
    ST_TO_ABOVE_CUP = 0
    ST_TO_GRASP = 1
    ST_CLOSE = 2
    ST_LIFT = 3
    ST_TO_ABOVE_COASTER = 4
    ST_TO_PLACE = 5
    ST_OPEN = 6
    ST_RETREAT = 7
    ST_TO_HOME = 8 
    ST_DONE = 9

    def __init__(self, cfg: CollectorCfg, env_cfg: Optional[CupOnCoasterEnvCfg] = None):
        self.cfg = cfg
        os.makedirs(cfg.save_dir, exist_ok=True)
        self.videos_dir = os.path.join(cfg.save_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

        # --- Env Setup ---
        self.env_cfg = env_cfg if env_cfg is not None else CupOnCoasterEnvCfg()
        self.env = CupOnCoasterEnv(self.env_cfg, num_envs=int(cfg.num_envs))
        self.env.reset()

        self.model = self.env.model
        self.B = int(cfg.num_envs)

        # Bodies
        self.cup_body = self.env.cup_body
        self.coaster_body = self.env.coaster_body

        # Cam view key
        self.cam_view_key = cfg.cam_view_key or "pixels/view_0"

        # --- Lifecycle ---
        self.active = np.zeros(self.B, dtype=bool)
        self.done = np.zeros(self.B, dtype=bool)
        self.success = np.zeros(self.B, dtype=bool)
        self.ctrl_step = np.zeros(self.B, dtype=np.int32)
        self._attempt_id = np.zeros(self.B, dtype=np.int64)

        # --- FSM State ---
        self.states = np.zeros(self.B, dtype=np.int32)
        self.state_enter_step = np.zeros(self.B, dtype=np.int32)
        self.place_hold_counter = np.zeros(self.B, dtype=np.int32)

        # Control Targets
        self.exec_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.exec_quat = np.zeros((self.B, 4), dtype=np.float32) # xyzw
        
        # Latch positions (关键：Env reset后会随机化位置，必须记录)
        self.latched_cup_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.latched_coaster_pos = np.zeros((self.B, 3), dtype=np.float32)

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
            "is_success": [],
            "video_frames": 0,
        }

    def start_episodes(self, env_ids: np.ndarray, seed: int) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if env_ids.size == 0:
            return

        # 1. Reset Env (Env 内部会处理随机化)
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
        self.place_hold_counter[env_ids] = 0

        # 3. Init Control Refs from Observation
        data = self.env._state.data
        self.env.robot.reset_envs(data, done_mask)
        self.env.robot.update_reference(data) 

        # 获取当前 EE Pose 作为初始控制目标，防止瞬间跳变
        all_poses = self.env.robot.get_ee_pose(data)
        for idx in env_ids:
            pose = all_poses[idx]
            self.exec_pos[idx] = pose[:3]
            if len(pose) == 7:
                self.exec_quat[idx] = pose[3:]
            elif len(pose) == 6:
                euler = pose[3:]
                self.exec_quat[idx] = Rotation.from_euler('xyz', euler).as_quat()

        # 4. Latch Logic: 获取当前随机化后的物体位置
        # 注意：这里我们使用 get_pose_batch 或直接循环读取
        for idx in env_ids:
            self.latched_cup_pos[idx] = self.env.cup_body.get_pose(data[idx])[:3]
            self.latched_coaster_pos[idx] = self.env.coaster_body.get_pose(data[idx])[:3]
        
        self.states[env_ids] = self.ST_TO_ABOVE_CUP

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
        
        is_success = bool(self.success[env_id])
        buf["is_success"].append(is_success)
        buf["reward"].append(float(1.0 if is_success else 0.0))

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
            try: os.remove(self._tmp_video_paths[env_id])
            except: pass
            
        self.buffers[env_id] = self._new_buffer()

    def _flush_jsonl(self, env_id: int, ep_idx: int, vid_path: str):
        path = os.path.join(self.cfg.save_dir, f"episode_{ep_idx:05d}.jsonl")
        buf = self.buffers[env_id]
        n = len(buf["times"])
        
        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                rec = {
                    "images_1": {"url": vid_path, "type": "video", "frame_idx": i},
                    "prompt": self.cfg.prompt,
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
        if not np.any(mask): return
        self.states[mask] = new_state
        self.state_enter_step[mask] = self.ctrl_step[mask].copy()

    def _step_logic(self) -> None:
        cfg = self.cfg
        B = self.B
        
        running = self.active & (~self.done)
        if not np.any(running): return

        cup_p = self.latched_cup_pos
        coaster_p = self.latched_coaster_pos
        
        # 1. 定义关键航点 (Waypoints)
        # 上方悬停
        p_above_cup = cup_p + np.array([0, 0, cfg.above_z])
        # 抓取位
        p_grasp     = cup_p + np.array([0, 0, cfg.grasp_down_z])
        # 抬起位
        p_lift      = cup_p + np.array([0, 0, cfg.lift_dz])
        # 杯垫上方悬停
        p_above_coaster = coaster_p + np.array([0, 0, cfg.above_z])
        # 放置位 (杯垫上方一点)
        p_place     = coaster_p + np.array([0, 0, cfg.place_z_offset])
        # 撤退位
        p_retreat   = coaster_p + np.array([0, 0, cfg.lift_dz * 2])
        # Home (桌面中心偏上)
        p_home      = np.tile(np.array([0.4, 0.0, 0.5], dtype=np.float32), (B, 1))

        # 2. 根据状态设定目标
        tgt_pos_curr = self.exec_pos.copy() # 默认保持
        grip_cmd = np.full((B,), cfg.gripper_open, dtype=np.float32)
        s = self.states

        # ST_TO_ABOVE_CUP
        mask = running & (s == self.ST_TO_ABOVE_CUP)
        if np.any(mask): tgt_pos_curr[mask] = p_above_cup[mask]
            
        # ST_TO_GRASP
        mask = running & (s == self.ST_TO_GRASP)
        if np.any(mask): tgt_pos_curr[mask] = p_grasp[mask]

        # ST_CLOSE (保持在抓取点，闭合夹爪)
        mask = running & (s == self.ST_CLOSE)
        if np.any(mask): 
            tgt_pos_curr[mask] = p_grasp[mask]
            grip_cmd[mask] = cfg.gripper_close

        # ST_LIFT (闭合夹爪，抬起)
        mask = running & (s == self.ST_LIFT)
        if np.any(mask):
            tgt_pos_curr[mask] = p_lift[mask]
            grip_cmd[mask] = cfg.gripper_close

        # ST_TO_ABOVE_COASTER (移动到杯垫上方)
        mask = running & (s == self.ST_TO_ABOVE_COASTER)
        if np.any(mask):
            tgt_pos_curr[mask] = p_above_coaster[mask]
            grip_cmd[mask] = cfg.gripper_close
            
        # ST_TO_PLACE (下降放置)
        mask = running & (s == self.ST_TO_PLACE)
        if np.any(mask):
            tgt_pos_curr[mask] = p_place[mask]
            grip_cmd[mask] = cfg.gripper_close
            
        # ST_OPEN (保持位置，张开夹爪)
        mask = running & (s == self.ST_OPEN)
        if np.any(mask):
            tgt_pos_curr[mask] = p_place[mask]
            grip_cmd[mask] = cfg.gripper_open
            
        # ST_RETREAT (向上撤退)
        mask = running & (s == self.ST_RETREAT)
        if np.any(mask):
            tgt_pos_curr[mask] = p_retreat[mask]
            grip_cmd[mask] = cfg.gripper_open

        # ST_TO_HOME
        mask = running & (s == self.ST_TO_HOME)
        if np.any(mask):
            tgt_pos_curr[mask] = p_home[mask]
            grip_cmd[mask] = cfg.gripper_open

        # 3. 平滑运动
        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos_curr, cfg.max_dp)
        
        # 4. 执行控制 (Simple P-Control for Relative Mode or Absolute Ref)
        ref_pose_6d = self.env.robot.ref_ee_pose 
        ref_pos = ref_pose_6d[:, :3]

        action = np.zeros((B, 7), dtype=np.float32)
        action[:, :3] = (self.exec_pos - ref_pos) * 0.5 
        action[:, 3:6] = 0 # 保持默认姿态 (向下)
        action[:, 6] = grip_cmd
        
        self._last_action[:] = action
        self.env.step(action)

        # 5. 状态跳转
        def is_reached(target_p):
            return np.linalg.norm(self.exec_pos - target_p, axis=1) < cfg.pos_tol

        # APPROACH -> GRASP
        mask = running & (s == self.ST_TO_ABOVE_CUP)
        if np.any(mask):
            self._enter_state(mask & is_reached(p_above_cup), self.ST_TO_GRASP)

        # GRASP -> CLOSE
        mask = running & (s == self.ST_TO_GRASP)
        if np.any(mask):
            self._enter_state(mask & is_reached(p_grasp), self.ST_CLOSE)
        
        # CLOSE -> LIFT (Wait for grip)
        mask = running & (s == self.ST_CLOSE)
        if np.any(mask):
            time_in_state = self.ctrl_step - self.state_enter_step
            # 可选：检查 grasp sensor
            closed_done = time_in_state >= cfg.close_hold_steps
            self._enter_state(mask & closed_done, self.ST_LIFT)
        
        # LIFT -> MOVE TO COASTER
        mask = running & (s == self.ST_LIFT)
        if np.any(mask):
            self._enter_state(mask & is_reached(p_lift), self.ST_TO_ABOVE_COASTER)
        
        # MOVE TO COASTER -> PLACE
        mask = running & (s == self.ST_TO_ABOVE_COASTER)
        if np.any(mask):
            self._enter_state(mask & is_reached(p_above_coaster), self.ST_TO_PLACE)

        # PLACE -> OPEN
        mask = running & (s == self.ST_TO_PLACE)
        if np.any(mask):
            self._enter_state(mask & is_reached(p_place), self.ST_OPEN)
        
        # OPEN -> RETREAT (Wait for release)
        mask = running & (s == self.ST_OPEN)
        if np.any(mask):
            m_open = mask
            # 使用计数器等待几步，确保物理稳定
            self.place_hold_counter[m_open] += 1
            ready_retreat = (self.place_hold_counter >= cfg.place_hold_steps)
            self._enter_state(m_open & ready_retreat, self.ST_RETREAT)
            
        # RETREAT -> HOME
        mask = running & (s == self.ST_RETREAT)
        if np.any(mask):
            self._enter_state(mask & is_reached(p_retreat), self.ST_TO_HOME)

        # HOME -> DONE
        mask = running & (s == self.ST_TO_HOME)
        if np.any(mask):
            self._enter_state(mask & is_reached(p_home), self.ST_DONE)

    # ----------------------------
    # Main Loop
    # ----------------------------
    def collect(self) -> None:
        cfg = self.cfg
        target_n = cfg.data_size
        
        all_ids = np.arange(self.B, dtype=np.int64)
        self.start_episodes(all_ids, seed=cfg.seed)
        
        print(f"Starting Cup-on-Coaster Collection. Target: {target_n}")
        
        while self.saved_count < target_n:
            self._step_logic()
            
            running = self.active & (~self.done)
            
            # Capture
            sample_mask = running & ((self.ctrl_step % cfg.sample_every_steps) == 0)
            for env_id in np.where(sample_mask)[0]:
                self._capture_step(env_id)
                
            # Render
            if cfg.save_video:
                render_mask = running & ((self.ctrl_step % cfg.render_every_steps) == 0)
                for env_id in np.where(render_mask)[0]:
                    self._write_video_frame(env_id)
            
            self.ctrl_step[running] += 1
            
            # Check Done / Timeout / Success
            for i in range(self.B):
                if not running[i]: continue
                
                fsm_done = (self.states[i] == self.ST_DONE)
                timeout = (self.ctrl_step[i] >= cfg.max_ctrl_steps)
                
                # 从 Env 的 info 中获取成功状态
                is_success_env = bool(self.env._state.info["is_success"][i])
                
                if fsm_done or timeout:
                    self.done[i] = True
                    # 只有当流程走完且环境判定为成功时，才算有效数据
                    self.success[i] = bool(fsm_done and is_success_env)
            
            # Handle Finished Episodes
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

    runner = CupOnCoasterCollector(cfg)
    try:
        runner.collect()
    except KeyboardInterrupt:
        pass
    finally:
        runner.close()

if __name__ == "__main__":
    main()