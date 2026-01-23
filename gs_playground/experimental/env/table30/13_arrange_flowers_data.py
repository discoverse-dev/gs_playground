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

# 导入你的插花环境
from gs_playground.src.manipulation.tasks.table30._13_arrange_flowers import (
    ArrangeFlowersEnv,
    ArrangeFlowersEnvCfg,
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def smooth_step_pos(curr: np.ndarray, tgt: np.ndarray, max_dp: float) -> np.ndarray:
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
        if bgr is None: return
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
    save_dir: str = "./data/table30_arrange_flowers_collect_manhattan"

    # env control
    max_ctrl_steps: int = 1000  # 增加步数以适应复杂动作

    # motion params
    max_dp: float = 0.005
    pos_tol: float = 0.002
    
    # rotation control
    rot_gain: float = 1.0     # 较高的增益以确保姿态锁定
    max_dr: float = 0.01      # 较慢的旋转速度以保持稳定
    angle_tol: float = 0.05   # 弧度容差

    # --- Task Specific Offsets ---
    grasp_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # 抬高高度 (绝对高度)
    lift_height_z: float = 0.45 # 稍微抬高一点，防止旋转时碰到瓶口
    
    # [参数] 分步旋转角度
    rot_x_deg: float = 60.0   # 第一步：绕Y轴旋转 90 度
    rot_z_deg: float = -35.0   # 第二步：绕Z轴旋转 30 度

    # [参数] 位置修正偏移 (Rotation Position Compensation)
    # 旋转后，需要平移一段距离让花对准瓶口。你可以在这里直接填入数值。
    align_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # [参数] 插入相关
    vase_rim_height: float = 0.55
    insert_depth: float = 0.15 # 插入深度

    # gripper
    gripper_open: float = 0.0
    gripper_close: float = 0.82

    # timing
    close_hold_steps: int = 20
    waypoint_dwell_steps: int = 15 # 稍微增加停留时间等待稳定

    # sampling / render
    sample_every_steps: int = 1
    save_video: bool = True
    render_every_steps: int = 1
    video_fps: int = 30
    video_w: int = 640
    video_h: int = 480
    cam_view_key: Optional[str] = "pixels/view_0"

    instruction: str = "Pick up the flower, move to vase, rotate Y then Z, align and insert."

# -----------------------------------------------------------------------------
# Collector
# -----------------------------------------------------------------------------
class ArrangeFlowersCollector:
    # --- FSM States ---

    # Phase 1: Approach & Grasp
    ST_APP_LIFT_Z = 0   
    ST_APP_ALIGN_X = 1  
    ST_APP_ALIGN_Y = 2  
    ST_APP_DESCEND = 3  
    ST_CLOSE = 4        

    # Phase 2: Transport (Move High)
    ST_LIFT_HIGH = 5    
    ST_TRP_ALIGN_X = 6  
    ST_TRP_ALIGN_Y = 7  

    # Phase 3: Fine Rotation (Two Steps)
    ST_ROT_X = 8     # 绕 Y 轴旋转 90 度
    ST_ROT_Z = 9     # 绕 Z 轴旋转 30 度

    # Phase 4: Align & Insert
    ST_ALIGN_POS = 10 # [New] 锁定角度，平移修正位置 (使用 align_offset)
    ST_INSERT = 11    # 向下插入
    
    # Phase 5: Release & Home
    ST_OPEN = 12        
    ST_RETREAT_Z = 13   
    ST_TO_HOME_X = 14   
    ST_TO_HOME_Y = 15   
    ST_TO_HOME_Z = 16   
    ST_DONE = 17

    def __init__(self, cfg: CollectorCfg, env_cfg: Optional[ArrangeFlowersEnvCfg] = None):
        self.cfg = cfg
        os.makedirs(cfg.save_dir, exist_ok=True)
        self.videos_dir = os.path.join(cfg.save_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

        # --- Env Setup ---
        self.env_cfg = env_cfg if env_cfg is not None else ArrangeFlowersEnvCfg()
        self.env = ArrangeFlowersEnv(self.env_cfg, num_envs=int(cfg.num_envs))
        self.env.reset()

        self.model = self.env.model
        self.B = int(cfg.num_envs)

        self.flower_body = self.env.flower_body
        self.vase_body = self.env.vase_body
        self.cam_view_key = cfg.cam_view_key or "pixels/view_0"
        
        # --- Lifecycle ---
        self.active = np.zeros(self.B, dtype=bool)
        self.done = np.zeros(self.B, dtype=bool)
        self.success = np.zeros(self.B, dtype=bool)
        self.ctrl_step = np.zeros(self.B, dtype=np.int32)

        # --- FSM State ---
        self.states = np.zeros(self.B, dtype=np.int32)
        self.state_enter_step = np.zeros(self.B, dtype=np.int32)
        self.state_reach_step = np.full(self.B, -1, dtype=np.int32)

        self._attempt_id = np.zeros(self.B, dtype=np.int64)

        # Control Targets
        self.exec_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.exec_quat = np.zeros((self.B, 4), dtype=np.float32)

        # Latch positions
        self.latched_flower_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.latched_vase_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.latched_start_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.latched_start_quat = np.zeros((self.B, 4), dtype=np.float32)
        
        # Rotation Targets
        self.quat_rot_y = np.zeros((self.B, 4), dtype=np.float32) # 第一步姿态
        self.quat_final = np.zeros((self.B, 4), dtype=np.float32) # 最终姿态

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
            "times": [], "logic_states": [], "qpos": [], "ee_pose": [],
            "gripper": [], "ctrl": [], "reward": [], "is_success": [], "video_frames": 0,
        }

    def start_episodes(self, env_ids: np.ndarray, seed: int) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if env_ids.size == 0: return

        try:
            self.env._rng = np.random.default_rng(int(seed))
        except Exception: pass

        done_mask = np.zeros((self.B,), dtype=bool)
        done_mask[env_ids] = True
        self.env.reset(done=done_mask)

        self.active[env_ids] = True
        self.done[env_ids] = False
        self.success[env_ids] = False
        self.ctrl_step[env_ids] = 0
        self.state_enter_step[env_ids] = 0
        self.state_reach_step[env_ids] = -1

        data = self.env._state.data
        self.env.robot.reset_envs(data, done_mask)
        self.env.robot.update_reference(data)

        # Initialize EE Pose
        all_poses = self.env.robot.get_ee_pose(data)
        for idx in env_ids:
            pose = all_poses[idx]
            self.exec_pos[idx] = pose[:3]
            self.latched_start_pos[idx] = pose[:3]
            if len(pose) == 7: self.exec_quat[idx] = pose[3:]
            elif len(pose) == 6:
                euler = pose[3:]
                self.exec_quat[idx] = Rotation.from_euler("xyz", euler).as_quat()
            self.latched_start_quat[idx] = self.exec_quat[idx].copy()

        # Latch Objects Poses
        flower_pose = np.stack([np.asarray(self.flower_body.get_pose(data), dtype=np.float32)], axis=1)[:, 0, :]
        vase_pose = np.stack([np.asarray(self.vase_body.get_pose(data), dtype=np.float32)], axis=1)[:, 0, :]
        
        self.latched_flower_pos[env_ids] = flower_pose[env_ids, :3]
        self.latched_vase_pos[env_ids] = vase_pose[env_ids, :3]

        # ---------------------------------------------------------------------
        # [Step-by-Step Rotation Calculation]
        # 1. 初始姿态 q_start
        # 2. 绕 Y 轴转 rot_y_deg -> q_rot_y
        # 3. 绕 Z 轴转 rot_z_deg -> q_final
        # ---------------------------------------------------------------------
        
        r_x = Rotation.from_euler('x', self.cfg.rot_x_deg, degrees=True)
        r_z = Rotation.from_euler('z', self.cfg.rot_z_deg, degrees=True)

        for idx in env_ids:
            q_start = Rotation.from_quat(self.latched_start_quat[idx])
            
            # Step 1: Rotate Y (Extrinsic World Y)
            # 先把手抬起来
            # q_rot_x = r_x * q_start
            reachable_euler = np.array([-155, 75, -60])
            q_reachable = Rotation.from_euler('xyz', reachable_euler, degrees=True)
            self.quat_rot_y[idx] = q_reachable.as_quat().astype(np.float32)
            # self.quat_rot_y[idx] = q_rot_x.as_quat().astype(np.float32)

            euler_deg = q_reachable.as_euler('xyz', degrees=True)
            # print(f"[Env {idx}] Step 1 (Rot Y) Euler XYZ: {euler_deg}")
            
            # Step 2: Rotate Z (Extrinsic World Z)
            # 再调整方向对准
            # q_final = r_z * q_rot_x
            q_final = r_z * q_reachable
            self.quat_final[idx] = q_final.as_quat().astype(np.float32)

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
            try: os.remove(tmp_path)
            except: pass
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
        buf["is_success"].append(bool(self.success[env_id]))
        buf["reward"].append(float(1.0 if self.success[env_id] else 0.0))

    def _write_video_frame(self, env_id: int) -> None:
        vw = self.video_writers[env_id]
        if vw is None: return
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
            try: os.remove(self._tmp_video_paths[env_id])
            except: pass
        self.buffers[env_id] = self._new_buffer()

    def _flush_jsonl(self, env_id: int, ep_idx: int, vid_path: str):
        path = os.path.join(self.cfg.save_dir, f"episode_{ep_idx:05d}.jsonl")
        buf = self.buffers[env_id]
        n = len(buf["times"])
        prompt = self.cfg.instruction
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

    def _enter_state(self, mask: np.ndarray, new_state: int) -> None:
        if not np.any(mask): return
        self.states[mask] = new_state
        self.state_enter_step[mask] = self.ctrl_step[mask].copy()
        self.state_reach_step[mask] = -1

    def _step_logic(self) -> None:
        cfg = self.cfg
        B = self.B
        obs = self.env._state.obs
        running = self.active & (~self.done)
        if not np.any(running): return
        
        s = self.states
        
        # --- 1. Define Key Points ---
        start_p = self.latched_start_pos
        flower_p = self.latched_flower_pos
        vase_p = self.latched_vase_pos
        g_off = np.array(cfg.grasp_offset, dtype=np.float32)
        align_off = np.array(cfg.align_offset, dtype=np.float32)
        
        # [Step 1] Approach Flower
        p_app_lift = start_p.copy()
        p_app_lift[:, 2] = np.maximum(start_p[:, 2], 0.3) 

        p_app_x = p_app_lift.copy()
        p_app_x[:, 0] = flower_p[:, 0] + g_off[0]

        p_app_y = p_app_x.copy()
        p_app_y[:, 1] = flower_p[:, 1] + g_off[1]

        p_grasp = p_app_y.copy()
        p_grasp[:, 2] = flower_p[:, 2] + g_off[2]

        # [Step 2] Lift High
        p_lift_high = p_grasp.copy()
        p_lift_high[:, 2] = cfg.lift_height_z

        # [Step 3] Approach Vase (High & XY Align)
        p_vase_hover = p_lift_high.copy()
        p_vase_hover[:, 0] = vase_p[:, 0]
        p_vase_hover[:, 1] = vase_p[:, 1]
        
        # [Step 4] Align Position (Using explicit offset)
        # 在旋转完成后，应用这个 Offset 来对准瓶口
        p_aligned = p_vase_hover.copy() 
        p_aligned += align_off 

        # [Step 5] Insert (Descend from Aligned)
        p_insert = p_aligned.copy()
        p_insert[:, 2] -= cfg.insert_depth

        # [Step 6] Retreat
        p_retreat = p_aligned.copy()
        p_retreat[:, 0] -= 0.2

        
        home_pos = np.tile(np.array([0.335, 0.0, 0.11], dtype=np.float32), (B, 1))
        p_home_x = p_retreat.copy(); p_home_x[:, 0] = home_pos[:, 0]
        p_home_y = p_home_x.copy(); p_home_y[:, 1] = home_pos[:, 1]
        p_home_z = home_pos.copy()

        # --- 2. Target Assignment ---
        tgt_pos_curr = self.exec_pos.copy()
        tgt_quat_curr = self.exec_quat.copy()
        grip_cmd = np.full((B,), cfg.gripper_open, dtype=np.float32)

        def set_target(state_id: int, pos: np.ndarray, quat: np.ndarray, grip: float):
            mask = running & (s == state_id)
            if np.any(mask):
                tgt_pos_curr[mask] = pos[mask]
                tgt_quat_curr[mask] = quat[mask]
                grip_cmd[mask] = float(grip)
        
        q_def = self.latched_start_quat
        q_step1 = self.quat_rot_y

        q_step2 = self.quat_final

        # Phase 1: Approach & Grasp
        set_target(self.ST_APP_LIFT_Z,  p_app_lift, q_def, cfg.gripper_open)
        set_target(self.ST_APP_ALIGN_X, p_app_x,    q_def, cfg.gripper_open)
        set_target(self.ST_APP_ALIGN_Y, p_app_y,    q_def, cfg.gripper_open)
        set_target(self.ST_APP_DESCEND, p_grasp,    q_def, cfg.gripper_open)
        set_target(self.ST_CLOSE,       p_grasp,    q_def, cfg.gripper_close)
        
        # Phase 2: Transport (High)
        set_target(self.ST_LIFT_HIGH,   p_lift_high, q_def, cfg.gripper_close)
        set_target(self.ST_TRP_ALIGN_X, p_vase_hover, q_def, cfg.gripper_close)
        set_target(self.ST_TRP_ALIGN_Y, p_vase_hover, q_def, cfg.gripper_close)
        
        # Phase 3: Fine Rotation (Two Steps, with Position Compliance)
        
        # Step 1: Rotate Y 90 deg
        set_target(self.ST_ROT_X,       p_vase_hover, q_step1, cfg.gripper_close)
        mask_rot_y = running & (s == self.ST_ROT_X)
        if np.any(mask_rot_y):
             # Position compliance: use actual position as target
             current_pos = obs["ee_pose"][mask_rot_y, :3]
             tgt_pos_curr[mask_rot_y] = current_pos
             self.exec_pos[mask_rot_y] = current_pos

        # Step 2: Rotate Z 30 deg
        set_target(self.ST_ROT_Z,       p_vase_hover, q_step2, cfg.gripper_close)
        mask_rot_z = running & (s == self.ST_ROT_Z)
        if np.any(mask_rot_z):
             # Position compliance
             current_pos = obs["ee_pose"][mask_rot_z, :3]
             tgt_pos_curr[mask_rot_z] = current_pos
             self.exec_pos[mask_rot_z] = current_pos
        
        # Phase 4: Align Position & Insert (Rotation Locked)
        # [Crucial Step] 旋转结束，恢复强力位置控制，并移动到 p_aligned
        set_target(self.ST_ALIGN_POS,   p_aligned,    q_step2, cfg.gripper_close)
        
        # Insert
        set_target(self.ST_INSERT,      p_insert,     q_step2, cfg.gripper_close)
        
        # Phase 5: Release & Home
        set_target(self.ST_OPEN,        p_insert,     q_step2, cfg.gripper_open)
        set_target(self.ST_RETREAT_Z,   p_retreat,    q_step2, cfg.gripper_open)
        set_target(self.ST_TO_HOME_X,   p_home_x,     q_def,   cfg.gripper_open)
        set_target(self.ST_TO_HOME_Y,   p_home_y,     q_def,   cfg.gripper_open)
        set_target(self.ST_TO_HOME_Z,   p_home_z,     q_def,   cfg.gripper_open)

        # --- 3. Execution ---
        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos_curr, cfg.max_dp)
        
        # Rotation Control
        for i in range(B):
            if not running[i]: continue
            r_curr = Rotation.from_quat(self.exec_quat[i])
            r_tgt = Rotation.from_quat(tgt_quat_curr[i])
            r_err = r_tgt * r_curr.inv()
            rotvec = r_err.as_rotvec()
            angle = np.linalg.norm(rotvec)
            if angle > 1e-6:
                step_angle = min(angle, cfg.max_dr)
                axis = rotvec / angle
                r_step = Rotation.from_rotvec(axis * step_angle)
                self.exec_quat[i] = (r_step * r_curr).as_quat()
            else:
                self.exec_quat[i] = tgt_quat_curr[i]
        # print("state",s)
        # Action (PD)
        ref_pose_6d = self.env.robot.ref_ee_pose
        ref_pos = ref_pose_6d[:, :3]
        ref_euler = ref_pose_6d[:, 3:6]
        ref_quat = Rotation.from_euler("xyz", ref_euler, degrees=False).as_quat().astype(np.float32)
        
        pos_err = (self.exec_pos - ref_pos) * 0.5 
        rotvec_cmd = np.zeros((B, 3), dtype=np.float32)
        for i in range(B):
            if not running[i]: continue
            r_des = Rotation.from_quat(self.exec_quat[i])
            r_ref = Rotation.from_quat(ref_quat[i])
            r_e = r_des * r_ref.inv()
            rv = r_e.as_rotvec()
            mag = np.linalg.norm(rv) + 1e-9
            scale = np.minimum(1.0, float(cfg.max_dr) / mag)
            rotvec_cmd[i] = rv * scale * cfg.rot_gain

        action = np.zeros((B, 7), dtype=np.float32)
        action[:, :3] = pos_err
        action[:, 3:6] = rotvec_cmd
        action[:, 6] = grip_cmd
        
        self._last_action[:] = action
        self.env.step(action)

        # --- 4. Transitions ---
        def is_pos_reached(target: np.ndarray) -> np.ndarray:
            return np.linalg.norm(self.exec_pos - target, axis=1) < cfg.pos_tol

        # 修改：增加 use_real_obs 选项，用于旋转检查
        def is_rot_reached(target_q: np.ndarray, use_real_obs: bool = False, debug_state: int = -1) -> np.ndarray:
            errs = []
            for i in range(B):
                # --- 获取当前姿态 ---
                if use_real_obs:
                     # 这里的 3: 取决于你的 obs 格式，确保取到的是四元数或欧拉角
                     rot_data = obs["ee_pose"][i, 3:] 
                     if len(rot_data) == 3:
                         r1 = Rotation.from_euler("xyz", rot_data, degrees=False)
                     elif len(rot_data) == 4:
                         r1 = Rotation.from_quat(rot_data)
                     else:
                         # 如果这里打印出来，说明 observation 数据维度不对
                         print(f"[ERROR] obs shape: {rot_data.shape}")
                         r1 = Rotation.identity()
                else:
                     curr_q = self.exec_quat[i]
                     r1 = Rotation.from_quat(curr_q)
                
                # --- 计算误差 ---
                r2 = Rotation.from_quat(target_q[i])
                dq = r1 * r2.inv()
                err_val = np.linalg.norm(dq.as_rotvec()) # 计算误差(弧度)
                errs.append(err_val)




            return np.array(errs) < cfg.angle_tol

        def _check_and_dwell(state_id: int, pos_tgt: np.ndarray, rot_tgt: np.ndarray, use_rot=False, check_real_rot=False) -> np.ndarray:
            in_state = running & (s == state_id)

            p_ok = is_pos_reached(pos_tgt)
            if check_real_rot :
                r_ok = is_rot_reached(rot_tgt, use_real_obs=check_real_rot) if use_rot else np.ones(B, dtype=bool)
                reached =  r_ok
            else :
                reached =  p_ok
            just_reached = in_state & reached & (self.state_reach_step == -1)
            if np.any(just_reached):
                self.state_reach_step[just_reached] = self.ctrl_step[just_reached]
            has_reached = (self.state_reach_step != -1)
            dwell_pass = (self.ctrl_step - self.state_reach_step) >= int(cfg.waypoint_dwell_steps)
            # if self.ctrl_step %50 ==0 :
            #     print("state",s)
            #     print("state_id",state_id)
            #     print(in_state,"in_state")
            #     print("has_reached",has_reached)
            #     print("dwell_pass",dwell_pass)
            #     print("reached",reached)
            return in_state & has_reached & dwell_pass & reached

        # 1. Approach
        self._enter_state(_check_and_dwell(self.ST_APP_LIFT_Z, p_app_lift, q_def), self.ST_APP_ALIGN_X)
        self._enter_state(_check_and_dwell(self.ST_APP_ALIGN_X, p_app_x, q_def), self.ST_APP_ALIGN_Y)
        self._enter_state(_check_and_dwell(self.ST_APP_ALIGN_Y, p_app_y, q_def), self.ST_APP_DESCEND)
        self._enter_state(_check_and_dwell(self.ST_APP_DESCEND, p_grasp, q_def), self.ST_CLOSE)
        
        # 2. Close
        mask_close = running & (s == self.ST_CLOSE)
        if np.any(mask_close):
            t_in = self.ctrl_step - self.state_enter_step
            done_close = t_in >= cfg.close_hold_steps
            self._enter_state(mask_close & done_close, self.ST_LIFT_HIGH)
            
        # 3. Lift -> Move to Vase X -> Y
        self._enter_state(_check_and_dwell(self.ST_LIFT_HIGH, p_lift_high, q_def), self.ST_TRP_ALIGN_X)
        self._enter_state(_check_and_dwell(self.ST_TRP_ALIGN_X, p_vase_hover, q_def), self.ST_TRP_ALIGN_Y)
        self._enter_state(_check_and_dwell(self.ST_TRP_ALIGN_Y, p_vase_hover, q_def), self.ST_ROT_X)
        
        # 4. Rotation Sequence (Pos Ignored)
        # Step 1: Rotate Y
        self._enter_state(_check_and_dwell(self.ST_ROT_X, self.exec_pos, q_step1, use_rot=True, check_real_rot=True), self.ST_ROT_Z)
        
        # Step 2: Rotate Z
        self._enter_state(_check_and_dwell(self.ST_ROT_Z, self.exec_pos, q_step2, use_rot=True, check_real_rot=True), self.ST_ALIGN_POS)
        
        # 5. Align Position (With Locked Rotation)
        self._enter_state(_check_and_dwell(self.ST_ALIGN_POS, p_aligned, q_step2, use_rot=True), self.ST_INSERT)
        
        # 6. Insert
        self._enter_state(_check_and_dwell(self.ST_INSERT, p_insert, q_step2, use_rot=True), self.ST_OPEN)
        
        # 7. Release & Home
        mask_open = running & (s == self.ST_OPEN)
        if np.any(mask_open):
            t_in = self.ctrl_step - self.state_enter_step
            done_open = t_in >= cfg.close_hold_steps
            self._enter_state(mask_open & done_open, self.ST_RETREAT_Z)
            
        self._enter_state(_check_and_dwell(self.ST_RETREAT_Z, p_retreat, q_step2), self.ST_TO_HOME_X)
        self._enter_state(_check_and_dwell(self.ST_TO_HOME_X, p_home_x, q_def), self.ST_TO_HOME_Y)
        self._enter_state(_check_and_dwell(self.ST_TO_HOME_Y, p_home_y, q_def), self.ST_TO_HOME_Z)
        self._enter_state(_check_and_dwell(self.ST_TO_HOME_Z, p_home_z, q_def), self.ST_DONE)

    def collect(self) -> None:
        cfg = self.cfg
        target_n = cfg.data_size
        all_ids = np.arange(self.B, dtype=np.int64)
        self.start_episodes(all_ids, seed=cfg.seed)

        print(f"Starting ArrangeFlowers Collection. Target: {target_n}")

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
                if not running[i]: continue
                fsm_done = (self.states[i] == self.ST_DONE)
                timeout = (self.ctrl_step[i] >= int(cfg.max_ctrl_steps))
                env_success = self.env.success_latched[i]

                if fsm_done or timeout:
                    self.done[i] = True
                    self.success[i] = bool(env_success) or bool(fsm_done)

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

    runner = ArrangeFlowersCollector(cfg)
    try:
        runner.collect()
    except KeyboardInterrupt:
        pass
    finally:
        runner.close()

if __name__ == "__main__":
    main()