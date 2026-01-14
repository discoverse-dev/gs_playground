from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from gs_playground.src.manipulation._tasks.common.cfg_base import BaseBatchCfg
from gs_playground.src.manipulation._tasks.common.runner_base import VectorizedTaskRunnerBase
from gs_playground.src.manipulation._tasks.common.episode import EpisodeIOCfg, EpisodeManager
from gs_playground.src.manipulation._tasks.common.jsonl import JsonlRecordSpec
from gs_playground.src.manipulation._tasks.common.motion import smooth_step_pos, get_pose_batch

from gs_playground.src.manipulation.tasks.table30._04_hang_toothbrush_cup import (
    HangBottleEnvCfg,
    HangBottleEnv,
)


@dataclass(frozen=True)
class HangBottleRunnerCfg(BaseBatchCfg):
    num_envs: int = 1
    action_mode: str = "eef"

    save_dir: str = "./data/table30_04_hang_bottle_ur5e_runner"
    subtask: str = "Hang the bottle on the rack."
    prompt: str = "hang bottle"

    # Motion params
    max_dp: float = 0.015 
    pos_tol: float = 0.015
    max_ctrl_steps: int = 1200 # 增加步数以容纳新的预备动作

    # Offsets (World Frame)
    grasp_offset: Tuple[float, float, float] = (0.0, -0.03, 0.0)
    
    # [新增] 预抓取高度偏移 (相对于抓取点向上 0.1m)
    pre_grasp_z: float = 0.10
    
    # 抬起高度
    lift_height: float = 0.20 
    # 预备挂载位置
    pre_hang_offset: Tuple[float, float, float] = (-0.047, -0.15, 0.03)
    # 挂载位置
    hang_offset: Tuple[float, float, float] = (-0.047, -0.069, 0.01)

    gripper_open: float = 0.0
    gripper_close: float = 0.82


class HangBottleRunner(VectorizedTaskRunnerBase):
    # [修改] 插入了 ST_GO_PRE_GRASP 状态
    ST_GO_PRE_GRASP = 0
    ST_GO_GRASP = 1
    ST_CLOSE = 2
    ST_LIFT = 3
    ST_GO_PRE_HANG = 4
    ST_HANG_DOWN = 5
    ST_RELEASE = 6
    ST_RETREAT = 7
    ST_DONE = 8

    def __init__(self, cfg: HangBottleRunnerCfg):
        self.cfg = cfg
        if int(cfg.num_envs) != 1:
            raise ValueError("Runner assumes num_envs=1.")

        super().__init__(
            batch_size=int(cfg.num_envs),
            data_size=int(cfg.data_size),
            seed=int(cfg.seed),
            log_every_s=float(cfg.log_every_s),
        )

        env_cfg = HangBottleEnvCfg(action_mode=str(cfg.action_mode))
        self.env = HangBottleEnv(env_cfg, num_envs=int(cfg.num_envs))
        self.env.reset()

        rec = JsonlRecordSpec(subtask=cfg.subtask, prompt=cfg.prompt, is_robot=True)
        io_cfg = EpisodeIOCfg(
            save_dir=str(cfg.save_dir),
            videos_subdir="videos",
            save_video=bool(cfg.save_video),
            video_fps=int(cfg.video_fps),
            video_w=int(cfg.video_w),
            video_h=int(cfg.video_h),
        )
        self.epio = EpisodeManager(batch_size=1, io_cfg=io_cfg, record_spec=rec)

        B = int(cfg.num_envs)
        self.states = np.zeros(B, dtype=np.int32)
        self.exec_target = np.zeros((B, 7), dtype=np.float32)
        self.exec_quat = np.zeros((B, 4), dtype=np.float32) 
        self._target_buf = np.zeros((B, 7), dtype=np.float32)

        # Latch
        self.latched_bottle_pos = np.zeros((B, 3), dtype=np.float32)
        self.latched_hook_pos = np.zeros((B, 3), dtype=np.float32)
        self.latched_grasp_quat = np.zeros((B, 4), dtype=np.float32)

        # Handles
        self.ee_site = self.env.robot.ee_site
        self.grasp_site = self.env.grasp_site
        self.hook_site = self.env.hook_site

    def _sample_every_steps(self) -> int: return max(1, self.cfg.sample_every_steps)
    def _render_every_steps(self) -> int: return max(1, self.cfg.render_every_steps)
    def _write_frame(self, env_id: int, bgr: Optional[np.ndarray]) -> None: self.epio.write_frame(env_id, bgr)

    def start_episode_env(self, env_id: int, seed_i: int) -> None:
        self.active[env_id] = True
        self.done[env_id] = False
        self.success[env_id] = False
        self.ctrl_step[env_id] = 0

        self.env.reset(done=np.array([True], dtype=bool))
        
        # [修改] 初始状态设为去预备点
        self.states[env_id] = self.ST_GO_PRE_GRASP
        
        data = self.env._state.data
        ee_pose = np.asarray(self.ee_site.get_pose(data[env_id]), dtype=np.float32).reshape(-1)
        self.exec_target[env_id] = ee_pose
        
        # Latch Positions
        bottle_p = np.asarray(self.grasp_site.get_pose(data[env_id]), dtype=np.float32)
        hook_p = np.asarray(self.hook_site.get_pose(data[env_id]), dtype=np.float32)
        
        self.latched_bottle_pos[env_id] = bottle_p[:3]
        self.latched_hook_pos[env_id] = hook_p[:3]
        self.latched_grasp_quat[env_id] = bottle_p[3:]

        # Init exec_quat
        self.exec_quat[env_id] = self.latched_grasp_quat[env_id]

        self.epio.reset_env(env_id)

    def step_task_vectorized(self) -> None:
        self._step_logic()

    def capture_env_step(self, env_id: int) -> None:
        obs = self.env._state.obs
        qpos = obs["qpos"][env_id].tolist()
        ee = obs["ee_pose"][env_id].tolist()
        grip = obs["gripper"][env_id].tolist()
        ctrl_vec = self._last_action[env_id].tolist() if hasattr(self, "_last_action") else []
        t_sec = float(self.ctrl_step[env_id] * float(self.env._cfg.ctrl_dt))

        self.epio.buffers[env_id].append(
            t=t_sec, logic_state=int(self.states[env_id]),
            qpos=qpos, ee_pose=ee, gripper=grip, ctrl=ctrl_vec,
        )

    def render_batch_bgr(self) -> List[Optional[np.ndarray]]:
        if not bool(self.cfg.save_video): return [None]
        img = self.env._state.obs["pixels/view_0"][0]
        return [img[..., ::-1].copy()]

    def is_env_terminal(self, env_id: int) -> bool:
        return bool(self.states[env_id] == self.ST_DONE or self.ctrl_step[env_id] >= self.cfg.max_ctrl_steps)

    def is_env_success(self, env_id: int) -> bool:
        return bool(self.states[env_id] == self.ST_DONE and self.env.is_hung[env_id])

    def finalize_env_hook(self, env_id: int) -> None:
        ep_idx = int(self.stats.saved_success)
        self.epio.finalize_env(env_id=env_id, episode_idx=ep_idx, success=bool(self.success[env_id]))

    def _step_logic(self) -> None:
        cfg = self.cfg
        B = 1
        data = self.env._state.data

        bottle_p = self.latched_bottle_pos.copy()
        hook_p = self.latched_hook_pos.copy()

        # --- Keypoints definition ---
        
        # 1. Grasp Target (Actual grasp)
        grasp_target = bottle_p + np.array(cfg.grasp_offset, dtype=np.float32)
        
        # 2. [新增] Pre-Grasp Target (Above bottle)
        pre_grasp_target = grasp_target.copy()
        pre_grasp_target[:, 2] += cfg.pre_grasp_z

        # 3. Lift Point
        lift_target = grasp_target.copy()
        lift_target[:, 2] += cfg.lift_height
        
        # 4. Hang Points
        pre_hang_target = hook_p + np.array(cfg.pre_hang_offset, dtype=np.float32)
        hang_target = hook_p + np.array(cfg.hang_offset, dtype=np.float32)
        retreat_target = hang_target.copy()
        retreat_target[:, 0] -= 0.1

        target = self._target_buf
        target[:] = self.exec_target
        
        s = self.states
        grip_cmd = float(cfg.gripper_open)

        # --- FSM Logic ---
        
        if s[0] == self.ST_GO_PRE_GRASP:
            # 移动到瓶子上方
            target[0, :3] = pre_grasp_target[0]
            grip_cmd = float(cfg.gripper_open)

        elif s[0] == self.ST_GO_GRASP:
            # 垂直下降
            target[0, :3] = grasp_target[0]
            grip_cmd = float(cfg.gripper_open)
            
        elif s[0] == self.ST_CLOSE:
            # 闭合夹爪
            target[0, :3] = grasp_target[0]
            grip_cmd = float(cfg.gripper_close)
            
        elif s[0] == self.ST_LIFT:
            # 垂直抬起
            target[0, :3] = lift_target[0]
            grip_cmd = float(cfg.gripper_close)
            
        elif s[0] == self.ST_GO_PRE_HANG:
            target[0, :3] = pre_hang_target[0]
            grip_cmd = float(cfg.gripper_close)
            
        elif s[0] == self.ST_HANG_DOWN:
            target[0, :3] = hang_target[0]
            grip_cmd = float(cfg.gripper_close)
            
        elif s[0] == self.ST_RELEASE:
            target[0, :3] = hang_target[0]
            grip_cmd = float(cfg.gripper_open)
            
        elif s[0] == self.ST_RETREAT:
            target[0, :3] = retreat_target[0]
            grip_cmd = float(cfg.gripper_open)

        # Apply Orientation
        target[:, 3:] = self.latched_grasp_quat

        # Smooth Motion
        self.exec_target[:, :3] = smooth_step_pos(self.exec_target[:, :3], target[:, :3], float(cfg.max_dp))
        
        # UR5e Adapter
        current_quat = self.exec_target[0, 3:]
        r = Rotation.from_quat(current_quat)
        current_rpy = r.as_euler('xyz', degrees=False)

        action = np.zeros((1, 7), dtype=np.float32)
        action[0, :3] = self.exec_target[0, :3]
        action[0, 3:6] = current_rpy
        action[0, 6] = grip_cmd

        self._last_action = action.copy()
        self.env.step(action)

        # --- Transitions ---
        ee_pose = get_pose_batch(self.ee_site, data, B)[:, :3]
        def reached(p, tol=cfg.pos_tol): return float(np.linalg.norm(ee_pose[0] - p[0])) < tol

        # FSM Transitions
        if s[0] == self.ST_GO_PRE_GRASP and reached(pre_grasp_target):
            self.states[0] = self.ST_GO_GRASP

        elif s[0] == self.ST_GO_GRASP and reached(grasp_target):
            self.states[0] = self.ST_CLOSE
            self.ctrl_step[0] = 0 # 重置计数器给 CLOSE 用
            
        elif s[0] == self.ST_CLOSE:
            # 等待夹爪闭合
            if self.ctrl_step[0] > 20: 
                self.states[0] = self.ST_LIFT
                
        elif s[0] == self.ST_LIFT and reached(lift_target):
            self.states[0] = self.ST_GO_PRE_HANG
            
        elif s[0] == self.ST_GO_PRE_HANG and reached(pre_hang_target):
            self.states[0] = self.ST_HANG_DOWN
            
        elif s[0] == self.ST_HANG_DOWN and reached(hang_target):
            self.states[0] = self.ST_RELEASE
            self.ctrl_step[0] = 0
            
        elif s[0] == self.ST_RELEASE:
            if self.ctrl_step[0] > 20:
                self.states[0] = self.ST_RETREAT
                
        elif s[0] == self.ST_RETREAT and reached(retreat_target):
            self.states[0] = self.ST_DONE


def main():
    cfg = HangBottleRunnerCfg(
        data_size=2,
        save_video=True,
        log_every_s=1.0,
    )
    runner = HangBottleRunner(cfg)
    runner.collect()
    print(f"[DONE] saved_success={runner.stats.saved_success}/{cfg.data_size}")
    print(f"Saved to: {cfg.save_dir}")

if __name__ == "__main__":
    main()