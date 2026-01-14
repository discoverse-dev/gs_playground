from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np
from scipy.spatial.transform import Rotation

from gs_playground.src.manipulation._tasks.common.cfg_base import BaseBatchCfg
from gs_playground.src.manipulation._tasks.common.runner_base import VectorizedTaskRunnerBase
from gs_playground.src.manipulation._tasks.common.episode import EpisodeIOCfg, EpisodeManager
from gs_playground.src.manipulation._tasks.common.jsonl import JsonlRecordSpec
from gs_playground.src.manipulation._tasks.common.motion import smooth_step_pos, get_pose_batch

# 引用新的 Env
from gs_playground.src.manipulation.tasks.table30._03_arrange_fruits_in_basket import (
    ArrangeFruitsEnvCfg,
    ArrangeFruitsEnv,
    FRUIT_NAMES,
)

@dataclass(frozen=True)
class StageOffsets:
    above_obj: Tuple[float, float, float]
    grasp: Tuple[float, float, float]
    lift: Tuple[float, float, float]
    above_container: Tuple[float, float, float]


@dataclass(frozen=True)
class ArrangeFruitsRunnerCfg(BaseBatchCfg):
    num_envs: int = 1
    action_mode: str = "eef"

    save_dir: str = "./data/table30_05_arrange_fruits_ur5e_runner"
    subtask: str = "Arrange fruits in basket."
    prompt: str = "arrange fruits"

    # Motion params
    max_dp: float = 0.005
    pos_tol: float = 0.011
    max_ctrl_steps: int = 1600

    gripper_open: float = 0.0
    gripper_close: float = 0.62

    # [默认配置]
    default_offsets: StageOffsets = StageOffsets(
        above_obj=(0.0, 0.0, 0.15),
        grasp=(0.0, 0.0, 0.01),
        lift=(0.0, 0.0, 0.20),
        above_container=(0.0, 0.0, 0.10),
    )
    
    obj_offsets: Dict[str, StageOffsets] = field(default_factory=lambda: {
        
        "fruit_avocado": StageOffsets(
            above_obj=(0.0, 0.0, 0.10), 
            grasp=(0.0, -0.05, -0.015), 
            lift=(0.0, 0.0, 0.20), 
            above_container=(0.0, 0.0, 0.10)
        ),

        "fruit_banana": StageOffsets(
            above_obj=(0.0, 0.0, 0.10), 
            grasp=(0.0, -0.05, -0.01), 
            lift=(0.0, 0.0, 0.20), 
            above_container=(0.0, 0.0, 0.25)
        ),

        "fruit_carambola": StageOffsets(
            above_obj=(0.0, 0.0, 0.10), 
            grasp=(-0.05, -0.03, 0.00), 
            lift=(0.0, 0.0, 0.20), 
            above_container=(0.0, 0.0, 0.25)
        ),

        "fruit_mangosteen": StageOffsets(
            above_obj=(0.0, 0.0, 0.10), 
            grasp=(-0.03, -0.02, -0.015), 
            lift=(0.0, 0.0, 0.20), 
            above_container=(0.0, 0.0, 0.25)
        ),
    })


class ArrangeFruitsRunner(VectorizedTaskRunnerBase):
    ST_IDLE = 0
    ST_MOVE_ABOVE = 1
    ST_DESCEND = 2
    ST_CLOSE = 3
    ST_LIFT = 4
    ST_MOVE_BASKET = 5
    ST_OPEN = 6
    ST_DONE = 7

    def __init__(self, cfg: ArrangeFruitsRunnerCfg):
        self.cfg = cfg
        if int(cfg.num_envs) != 1:
            raise ValueError("Runner assumes num_envs=1.")

        super().__init__(
            batch_size=int(cfg.num_envs),
            data_size=int(cfg.data_size),
            seed=int(cfg.seed),
            log_every_s=float(cfg.log_every_s),
        )

        env_cfg = ArrangeFruitsEnvCfg(action_mode=str(cfg.action_mode))
        self.env = ArrangeFruitsEnv(env_cfg, num_envs=int(cfg.num_envs))
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

        # Handles
        self.ee_site = self.env.robot.ee_site
        self.basket_site = self.env.basket_site
        self.fruit_bodies = self.env.fruit_bodies
        
        # Logic State
        self.current_fruit_idx = np.zeros(B, dtype=np.int32)
        self.latched_fruit_pos = np.zeros((B, 3), dtype=np.float32)
        self.latched_fruit_quat = np.zeros((B, 4), dtype=np.float32)
        self.state_step = np.zeros(B, dtype=np.int32)

    def _sample_every_steps(self) -> int: return max(1, self.cfg.sample_every_steps)
    def _render_every_steps(self) -> int: return max(1, self.cfg.render_every_steps)
    def _write_frame(self, env_id: int, bgr: Optional[np.ndarray]) -> None: self.epio.write_frame(env_id, bgr)

    def start_episode_env(self, env_id: int, seed_i: int) -> None:
        self.active[env_id] = True
        self.done[env_id] = False
        self.success[env_id] = False
        self.ctrl_step[env_id] = 0
        self.state_step[env_id] = 0

        self.env.reset(done=np.array([True], dtype=bool))
        
        self.states[env_id] = self.ST_IDLE
        self.current_fruit_idx[env_id] = 0
        
        data = self.env._state.data
        ee_pose = np.asarray(self.ee_site.get_pose(data[env_id]), dtype=np.float32).reshape(-1)
        self.exec_target[env_id] = ee_pose
        self.exec_quat[env_id] = ee_pose[3:7] 

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
        return bool(self.states[env_id] == self.ST_DONE)

    def finalize_env_hook(self, env_id: int) -> None:
        ep_idx = int(self.stats.saved_success)
        self.epio.finalize_env(env_id=env_id, episode_idx=ep_idx, success=bool(self.success[env_id]))

    def _get_stage_offsets(self, fruit_idx: int) -> StageOffsets:
        name = FRUIT_NAMES[fruit_idx]
        return self.cfg.obj_offsets.get(name, self.cfg.default_offsets)

    def _step_logic(self) -> None:
        cfg = self.cfg
        B = 1
        data = self.env._state.data
        self.state_step[0] += 1
        
        env_target_idx = int(self.env.current_obj_idx[0])
        
        if env_target_idx >= len(FRUIT_NAMES):
            self.states[0] = self.ST_DONE
        elif env_target_idx > self.current_fruit_idx[0]:
            self.current_fruit_idx[0] = env_target_idx
            self.states[0] = self.ST_IDLE

        cur_idx = self.current_fruit_idx[0]
        s = self.states[0]
        if self.ctrl_step[0] % 50 == 0:
            print(f"[DEBUG] Env=0, State={s}, Fruit={cur_idx}, Step={self.ctrl_step[0]}")
        
        # Latch
        if s == self.ST_IDLE and cur_idx < len(FRUIT_NAMES):
            fruit_p = np.asarray(self.fruit_bodies[cur_idx].get_pose(data[0]), dtype=np.float32)
            self.latched_fruit_pos[0] = fruit_p[:3]
            self.latched_fruit_quat[0] = self.exec_quat[0] 
            self.states[0] = self.ST_MOVE_ABOVE
        
        # Targets
        offsets = self._get_stage_offsets(cur_idx)
        fruit_p = self.latched_fruit_pos[0]
        basket_p = np.asarray(self.basket_site.get_pose(data[0]), dtype=np.float32)[:3]
        
        above_target = fruit_p + np.array(offsets.above_obj, dtype=np.float32)
        grasp_target = fruit_p + np.array(offsets.grasp, dtype=np.float32)
        lift_target = grasp_target + np.array(offsets.lift, dtype=np.float32)
        
        basket_target = basket_p + np.array(offsets.above_container, dtype=np.float32)
        
        target = self._target_buf
        target[:] = self.exec_target
        grip_cmd = float(cfg.gripper_open)

        # FSM
        if s == self.ST_MOVE_ABOVE:
            target[0, :3] = above_target
            grip_cmd = float(cfg.gripper_open)
            
        elif s == self.ST_DESCEND:
            target[0, :3] = grasp_target
            grip_cmd = float(cfg.gripper_open)
            
        elif s == self.ST_CLOSE:
            target[0, :3] = grasp_target
            grip_cmd = float(cfg.gripper_close)
            
        elif s == self.ST_LIFT:
            target[0, :3] = lift_target
            grip_cmd = float(cfg.gripper_close)
            
        elif s == self.ST_MOVE_BASKET:
            target[0, :3] = basket_target
            grip_cmd = float(cfg.gripper_close)
            
        elif s == self.ST_OPEN:
            target[0, :3] = basket_target
            grip_cmd = float(cfg.gripper_open)

        target[:, 3:] = self.latched_fruit_quat[0]

        # Smooth
        self.exec_target[:, :3] = smooth_step_pos(self.exec_target[:, :3], target[:, :3], float(cfg.max_dp))
        
        # Rotation
        current_quat = self.exec_target[0, 3:]
        r = Rotation.from_quat(current_quat)
        current_rpy = r.as_euler('xyz', degrees=False)

        action = np.zeros((1, 7), dtype=np.float32)
        action[0, :3] = self.exec_target[0, :3]
        action[0, 3:6] = current_rpy
        action[0, 6] = grip_cmd

        self._last_action = action.copy()
        self.env.step(action)

        # Transitions
        ee_pose = get_pose_batch(self.ee_site, data, B)[:, :3]
        def reached(p, tol=cfg.pos_tol): return float(np.linalg.norm(ee_pose[0] - p)) < tol
        print(float(np.linalg.norm(ee_pose[0] - basket_target)))
        if s == self.ST_MOVE_ABOVE and reached(above_target):
            self.states[0] = self.ST_DESCEND
            
        elif s == self.ST_DESCEND and reached(grasp_target):
            self.states[0] = self.ST_CLOSE
            self.state_step[0] = 0
            
        elif s == self.ST_CLOSE:
            if self.ctrl_step[0] > 15:
                self.states[0] = self.ST_LIFT
                
        elif s == self.ST_LIFT and reached(lift_target):
            self.states[0] = self.ST_MOVE_BASKET
            
        elif s == self.ST_MOVE_BASKET and reached(basket_target):
            self.states[0] = self.ST_OPEN
            self.state_step[0] = 0
            
        elif s == self.ST_OPEN:
            if self.ctrl_step[0] > 15:
                self.states[0] = self.ST_IDLE


def main():
    cfg = ArrangeFruitsRunnerCfg(
        data_size=1,
        save_video=True,
        log_every_s=1.0,
    )
    runner = ArrangeFruitsRunner(cfg)
    runner.collect()
    print(f"[DONE] saved_success={runner.stats.saved_success}/{cfg.data_size}")
    print(f"Saved to: {cfg.save_dir}")

if __name__ == "__main__":
    main()