from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from scipy.spatial.transform import Rotation

from gs_playground.src.manipulation._tasks.common.cfg_base import BaseBatchCfg
from gs_playground.src.manipulation._tasks.common.runner_base import VectorizedTaskRunnerBase
from gs_playground.src.manipulation._tasks.common.episode import EpisodeIOCfg, EpisodeManager
from gs_playground.src.manipulation._tasks.common.jsonl import JsonlRecordSpec
from gs_playground.src.manipulation._tasks.common.motion import smooth_step_pos, get_pose_batch

from gs_playground.src.manipulation.tasks.table30._02_stack_color_blocks import (
    StackColorBlocksEnvCfg,
    StackColorBlocksEnv,
)


@dataclass(frozen=True)
class StackColorBlocksRunnerCfg(BaseBatchCfg):
    num_envs: int = 1
    # [修复1] 必须使用 "eef" 模式，对应 BaseRobot 的 Cartesian 控制
    action_mode: str = "eef"

    # dataset
    save_dir: str = "./data/table30_02_stack_color_blocks_ur5e_env_runner"

    # task text
    subtask: str = "Select any two blocks from the three blocks on the table and stack one on top of the other."
    prompt: str = "stack cubes"

    # motion / fsm
    max_dp: float = 0.02
    pos_tol: float = 0.02
    max_ctrl_steps: int = 600  # 稍微增加一点步数以防万一

    above_z: float = 0.18
    # [调整] 抓取高度偏移，0.0 表示抓取物体中心
    grasp_down_z: float = 0.00
    lift_dz: float = 0.10

    cube_half: float = 0.026
    stack_xy_tol: float = 0.030
    stack_z_tol: float = 0.06
    stack_hold_k: int = 10

    gripper_close: float = 0.82
    gripper_open: float = 0.0


class StackColorBlocksEnvRunner(VectorizedTaskRunnerBase):
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

    def __init__(self, cfg: StackColorBlocksRunnerCfg):
        self.cfg = cfg
        if int(cfg.num_envs) != 1:
            raise ValueError("Runner currently assumes num_envs=1.")

        super().__init__(
            batch_size=int(cfg.num_envs),
            data_size=int(cfg.data_size),
            seed=int(cfg.seed),
            log_every_s=float(cfg.log_every_s),
        )

        env_cfg = StackColorBlocksEnvCfg(action_mode=str(cfg.action_mode))
        self.env = StackColorBlocksEnv(env_cfg, num_envs=int(cfg.num_envs))
        self.env.reset()

        # episode io
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

        # state
        self.states = np.zeros(1, dtype=np.int32)
        self.exec_target = np.zeros((1, 7), dtype=np.float32)
        self.exec_quat = np.zeros((1, 4), dtype=np.float32)
        self.top_idx = np.zeros(1, dtype=np.int32)
        self.base_idx = np.zeros(1, dtype=np.int32)
        self.stack_hold = np.zeros(1, dtype=np.int32)

        self._target_buf = np.zeros((1, 7), dtype=np.float32)
        self._grip_buf = np.zeros(1, dtype=np.float32)

        # [修复2] 初始化锁存变量，用于记录抓取前的物体位置
        self.latched_top_pos = np.zeros((1, 3), dtype=np.float32)
        self.latched_base_pos = np.zeros((1, 3), dtype=np.float32)

        # handles
        self.ee_site = self.env.model.get_site("gripper")
        self.cube_bodies = self.env.cube_bodies

    def _sample_every_steps(self) -> int:
        return int(max(1, self.cfg.sample_every_steps))

    def _render_every_steps(self) -> int:
        return int(max(1, self.cfg.render_every_steps))

    def _write_frame(self, env_id: int, bgr: Optional[np.ndarray]) -> None:
        self.epio.write_frame(env_id, bgr)

    def start_episode_env(self, env_id: int, seed_i: int) -> None:
        rng = np.random.RandomState(int(seed_i))

        self.active[env_id] = True
        self.done[env_id] = False
        self.success[env_id] = False
        self.ctrl_step[env_id] = 0

        self.env.reset(done=np.array([True], dtype=bool))

        perm = rng.permutation(3)
        self.top_idx[env_id] = int(perm[0])
        self.base_idx[env_id] = int(perm[1])
        self.stack_hold[env_id] = 0

        data = self.env._state.data
        ee_pose = np.asarray(self.ee_site.get_pose(data[env_id]), dtype=np.float32).reshape(-1)
        self.exec_target[env_id] = ee_pose
        self.exec_quat[env_id] = ee_pose[3:7]

        self.states[env_id] = self.ST_SAMPLE_PAIR
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
            t=t_sec,
            logic_state=int(self.states[env_id]),
            qpos=qpos,
            ee_pose=ee,
            gripper=grip,
            ctrl=ctrl_vec,
        )

    def render_batch_bgr(self) -> List[Optional[np.ndarray]]:
        if not bool(self.cfg.save_video):
            return [None]
        img = self.env._state.obs["pixels/view_0"][0]  # RGB
        return [img[..., ::-1].copy()]  # BGR

    def is_env_terminal(self, env_id: int) -> bool:
        return bool(self.states[env_id] == self.ST_DONE or self.ctrl_step[env_id] >= self.cfg.max_ctrl_steps)

    def is_env_success(self, env_id: int) -> bool:
        return bool(self.states[env_id] == self.ST_DONE and self._stack_success(env_id))

    def finalize_env_hook(self, env_id: int) -> None:
        ep_idx = int(self.stats.saved_success)
        self.epio.finalize_env(env_id=env_id, episode_idx=ep_idx, success=bool(self.success[env_id]))

    def _stack_success(self, env_id: int) -> bool:
        cfg = self.cfg
        top_i = int(self.top_idx[env_id])
        base_i = int(self.base_idx[env_id])
        data = self.env._state.data

        top_pose = np.asarray(self.cube_bodies[top_i].get_pose(data[env_id]), dtype=np.float32).reshape(-1)
        base_pose = np.asarray(self.cube_bodies[base_i].get_pose(data[env_id]), dtype=np.float32).reshape(-1)

        tp, bp = top_pose[:3], base_pose[:3]
        xy_ok = float(np.linalg.norm(tp[:2] - bp[:2])) < float(cfg.stack_xy_tol)
        z_target = float(bp[2] + 2.0 * float(cfg.cube_half))
        z_ok = abs(float(tp[2] - z_target)) < float(cfg.stack_z_tol)
        return bool(xy_ok and z_ok)

    def _step_logic(self) -> None:
        cfg = self.cfg
        B = 1

        data = self.env._state.data
        cube_pose = np.stack([get_pose_batch(b, data, B) for b in self.cube_bodies], axis=1)  # (1,3,7)
        cube_p = cube_pose[:, :, :3]

        top_i = np.clip(self.top_idx, 0, 2)
        base_i = np.clip(self.base_idx, 0, 2)
        
        # [修复3] 状态机开始时，立刻记录(Latch)方块的位置。
        # 这样即使方块被抓起来了，top_p 依然是指向桌面的原始位置。
        s = int(self.states[0])
        if s == self.ST_SAMPLE_PAIR:
            self.latched_top_pos[0] = cube_p[0, top_i[0], :]
            self.latched_base_pos[0] = cube_p[0, base_i[0], :]
            
            # 状态跳转
            self.states[0] = self.ST_TO_ABOVE_TOP
            s = int(self.states[0])

        # 使用锁存的静态位置进行计算
        top_p = self.latched_top_pos.copy()
        base_p = self.latched_base_pos.copy()

        above_top = top_p.copy();  above_top[:, 2] += float(cfg.above_z)
        grasp_top = top_p.copy();  grasp_top[:, 2] += float(cfg.grasp_down_z)
        lift_p    = above_top.copy(); lift_p[:, 2] += float(cfg.lift_dz)
        above_base = base_p.copy(); above_base[:, 2] += float(cfg.above_z + 0.05)
        stack_p    = base_p.copy(); stack_p[:, 2] += float(2.0 * cfg.cube_half + 0.005)
        retreat_p  = above_base.copy(); retreat_p[:, 2] += float(cfg.lift_dz)

        target = self._target_buf
        grip = self._grip_buf
        target[:] = self.exec_target
        grip[:] = float(cfg.gripper_open)

        # 状态机逻辑
        if s == self.ST_TO_ABOVE_TOP:
            target[0, :3] = above_top[0]
        elif s == self.ST_TO_GRASP:
            target[0, :3] = grasp_top[0]
        elif s == self.ST_CLOSE:
            grip[0] = float(cfg.gripper_close)
            target[0, :3] = grasp_top[0]
        elif s == self.ST_LIFT:
            grip[0] = float(cfg.gripper_close)
            target[0, :3] = lift_p[0] # lift_p 现在是固定的
        elif s == self.ST_TO_ABOVE_BASE:
            grip[0] = float(cfg.gripper_close)
            target[0, :3] = above_base[0]
        elif s == self.ST_TO_STACK:
            grip[0] = float(cfg.gripper_close)
            target[0, :3] = stack_p[0]
        elif s == self.ST_OPEN_HOLD:
            grip[0] = float(cfg.gripper_open)
            target[0, :3] = stack_p[0]
        elif s == self.ST_RETREAT:
            grip[0] = float(cfg.gripper_open)
            target[0, :3] = retreat_p[0]

        # 姿态保持
        target[0, 3:] = self.exec_quat[0]

        # 运动平滑
        self.exec_target[:, :3] = smooth_step_pos(self.exec_target[:, :3], target[:, :3], float(cfg.max_dp))
        self.exec_target[:, 3:] = target[:, 3:]

        # [修复4] 构建 Action：将四元数转换为 Euler RPY
        current_quat = self.exec_target[0, 3:] # [x, y, z, w]
        r = Rotation.from_quat(current_quat)
        # BaseRobot 内部逻辑需要 [roll, pitch, yaw] 顺序
        current_rpy = r.as_euler('xyz', degrees=False)

        # 构造最终的 7维 Action: [x,y,z, r,p,y, gripper]
        action = np.zeros((1, 7), dtype=np.float32)
        action[0, :3] = self.exec_target[0, :3] 
        action[0, 3:6] = current_rpy            
        action[0, 6] = grip[0]                  

        self._last_action = action.copy()
        
        # 发送 Action
        self.env.step(action)

        # [判定逻辑] Reach Checks
        ee_pose2 = get_pose_batch(self.ee_site, self.env._state.data, B)
        ee_p2 = ee_pose2[:, :3]

        def reached(p): return float(np.linalg.norm(ee_p2[0] - p[0])) < float(cfg.pos_tol)

        if self.states[0] == self.ST_TO_ABOVE_TOP and reached(above_top):
            self.states[0] = self.ST_TO_GRASP
        elif self.states[0] == self.ST_TO_GRASP and reached(grasp_top):
            self.states[0] = self.ST_CLOSE
        elif self.states[0] == self.ST_CLOSE and (self.ctrl_step[0] % 15 == 0): # 稍微增加一点抓取等待时间
            self.states[0] = self.ST_LIFT
        elif self.states[0] == self.ST_LIFT and reached(lift_p):
            self.states[0] = self.ST_TO_ABOVE_BASE
        elif self.states[0] == self.ST_TO_ABOVE_BASE and reached(above_base):
            self.states[0] = self.ST_TO_STACK
        elif self.states[0] == self.ST_TO_STACK and reached(stack_p):
            self.states[0] = self.ST_OPEN_HOLD
            self.stack_hold[0] = 0
        elif self.states[0] == self.ST_OPEN_HOLD:
            if self._stack_success(0):
                self.stack_hold[0] += 1
            else:
                self.stack_hold[0] = 0
            if self.stack_hold[0] >= int(cfg.stack_hold_k):
                self.states[0] = self.ST_RETREAT
        elif self.states[0] == self.ST_RETREAT and reached(retreat_p):
            self.states[0] = self.ST_DONE


def main():
    cfg = StackColorBlocksRunnerCfg(
        data_size=2,
        save_video=True,
        log_every_s=1.0,
    )
    print("[DBG] build runner")
    runner = StackColorBlocksEnvRunner(cfg)
    print("[DBG] collect")
    runner.collect()
    print(f"[DONE] saved_success={runner.stats.saved_success}/{cfg.data_size}, attempted={runner.stats.attempted}")
    print(f"Saved to: {cfg.save_dir}")


if __name__ == "__main__":
    main()