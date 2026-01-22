from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import motrixsim as mtx
import numpy as np
from motrixsim import SceneData
import numpy as np
from scipy.spatial.transform import Rotation as R
from gs_playground import ROOT_PATH
from gs_playground.src.env.registry import envcfg, env
from gs_playground.src.manipulation.tasks.task_env import TaskEnvCfg, TaskEnv
from gs_playground.src.env.motrix_env.render_env import RenderEnvState

ASSETS_TASK_DIR = ROOT_PATH / "models" / "tasks" / "table30" / "_02_stack_color_blocks" / "3dgs"
TASK_GAUSSIANS = {
    "cube_blue"   : ASSETS_TASK_DIR /  "cube_blue.ply",
    "cube_yellow" : ASSETS_TASK_DIR / "cube_yellow.ply",
    "cube_orange" : ASSETS_TASK_DIR / "cube_orange.ply",
}

@envcfg("table30/stack_color_blocks_franka")
@dataclass
class StackColorBlocksEnvCfg(TaskEnvCfg):
    # model / sim
    robot_name: str = "franka_robotiq"
    model_file: str = str((ROOT_PATH / "models" / "robots" / 
                           "manipulation" / "franka_emika_panda_robotiq" / 
                           "xmls" / "table30_02_stack_color_blocks.xml").as_posix())

    # control
    action_mode: str = "eef_relative"  # "joint" or "eef"

    # rendering
    img_width: int = 640
    img_height: int = 480
 
    # observation / prompt
    instruction: str = "Stack the yellow block on top of the orange block."

    # task entities
    cube_names: Tuple[str, str, str] = ("cube_blue", "cube_yellow", "cube_orange")
    
    # task params
    success_dist_xy: float = 0.05
    success_delta_z_min: float = 0.02
    success_delta_z_max: float = 0.10
    gripper_close_thresh: float = 0.2
    grasp_dist_thresh: float = 0.03


@env("table30/stack_color_blocks_franka", "np")
class StackColorBlocksEnv(TaskEnv):
    """
    Task: Stack color blocks.
    Robot: UR5e + Robotiq 2F-85.
    Backend: MotrixSim (np).
    """

    def __init__(self, cfg: StackColorBlocksEnvCfg, num_envs: int = 32):
        super().__init__(cfg, num_envs=num_envs)

        self.cube_bodies = [self.model.get_body(self.model.get_body_index(n)) for n in cfg.cube_names]


        self.top_idx = np.zeros((self.num_envs,), dtype=np.int32)
        self.base_idx = np.zeros((self.num_envs,), dtype=np.int32)
        self.grasp_latched = np.zeros((self.num_envs,), dtype=bool)
        self.success_latched = np.zeros((self.num_envs,), dtype=bool)

    # ---- Task hooks ----
    def task_gaussians(self) -> Dict[str, str]:
        return TASK_GAUSSIANS


    # -----------------------------------------------------------------------------
    # In your env task file: StackColorBlocksEnv._randomize
    # Change yaw sampling range from [-pi, pi] to [-pi/2, pi/2].
    # -----------------------------------------------------------------------------

    def _randomize(self, data: SceneData, done_mask: np.ndarray, phase: str = "reset"):
            if data.shape[0] == 0:
                return



            B = data.shape[0]
            C = len(self.cube_bodies)

            min_xy_dist = 0.08
            max_tries = 100  # 增加尝试次数以确保在有限空间内能找到不重叠的解

            # ---- XML <geom name="range"> 参数 ----
            # pos="0.5 0.02 0.055", size="0.125 0.225 0.001"
            range_center = np.array([0.42, 0.02], dtype=np.float32)
            range_half_size = np.array([0.125, 0.225], dtype=np.float32)
            
            # 计算采样边界
            lower_bound = range_center - range_half_size
            upper_bound = range_center + range_half_size

            # 获取当前位姿作为基础容器 (主要用于保持 Z 轴高度和初始旋转)
            cube_pose = np.stack(
                [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.cube_bodies],
                axis=1,
            )
            new_pose = cube_pose.copy()
            
            # 初始化采样状态
            # remaining: 标记哪些环境(env)还没有生成合法的(不重叠的)位置
            remaining = np.ones((B,), dtype=bool)
            
            # 用于距离计算的对角线掩码 (避免计算自己到自己的距离为0)
            eye_mask = (np.eye(C, dtype=np.float32) * 1e6)[None, :, :]

            # ---- 1) 在 range 范围内随机生成 XY，并保证最小间距 ----
            for _ in range(max_tries):
                if not remaining.any():
                    break

                n_rem = int(remaining.sum())
                
                # 在边界内均匀采样: shape (n_rem, C, 2)
                # cand_xy = lower + (upper - lower) * rand
                rand_uniform = self._rng.uniform(0.0, 1.0, size=(n_rem, C, 2)).astype(np.float32)
                cand_xy = lower_bound + (upper_bound - lower_bound) * rand_uniform

                # 计算两两之间的距离
                # diff shape: (n_rem, C, C, 2)
                diff = cand_xy[:, :, None, :] - cand_xy[:, None, :, :]
                # dist shape: (n_rem, C, C)
                dist = np.linalg.norm(diff, axis=-1) + eye_mask
                
                # 检查是否所有物体的间距都 >= min_xy_dist
                # ok shape: (n_rem,)
                ok = dist.min(axis=(1, 2)) >= float(min_xy_dist)

                if ok.any():
                    rem_idx = np.where(remaining)[0] # 找出所有还需要处理的 env 索引
                    good_envs = rem_idx[ok]          # 从中找出这一轮生成成功的 env 索引
                    
                    # 将成功的 XY 坐标填入 new_pose
                    new_pose[good_envs, :, :2] = cand_xy[ok]
                    
                    # 标记这些环境已完成
                    remaining[good_envs] = False

            # ---- 2) 添加随机 Yaw 旋转 (保持 [-90, 90] 度) ----
            yaw = self._rng.uniform(-0.25 * np.pi, 0.25 * np.pi, size=(B, C)).astype(np.float32)

            # 假设四元数顺序为 xyzw (scipy 默认)
            q_old = new_pose[..., 3:7].reshape(-1, 4) 
            # print("q_old",q_old)
            r_old = R.from_quat(q_old)
            r_yaw = R.from_euler("z", yaw.reshape(-1), degrees=False)
            r_new = r_yaw * r_old

            q_new = r_new.as_quat().astype(np.float32).reshape(B, C, 4)
            # print("q_new",q_new)

            new_pose[..., 3:7] = q_new

            # ---- 3) 随机打乱方块索引 (Permute) ----
            # 这一步让颜色和位置的对应关系随机化
            perm = np.argsort(self._rng.random((B, C)).astype(np.float32), axis=1)
            new_pose = np.take_along_axis(new_pose, perm[:, :, None], axis=1)

            # ---- 写回物理引擎 ----
            for env_idx in range(B):
                for cube_idx, body in enumerate(self.cube_bodies):
                    body.set_dof_pos(
                        data[env_idx],
                        new_pose[env_idx, cube_idx],
                        include_floatingbase=True,
                    )


    def _reset_task_state(self, done: np.ndarray):
        """Reset internal task state variables for done environments."""
        n_done = np.sum(done)
        if n_done > 0:

            try:
                yellow_idx = self._cfg.cube_names.index("cube_yellow")
                orange_idx = self._cfg.cube_names.index("cube_orange")
            except ValueError:
                # 如果名字写错了，回退到默认索引 1 和 2
                yellow_idx = 1
                orange_idx = 2

            # 2. 强制赋值：所有 reset 的环境，目标都是 黄色(top) -> 橙色(base)
            self.top_idx[done] = yellow_idx
            self.base_idx[done] = orange_idx

            # 3. 重置状态锁存
            self.grasp_latched[done] = False
            self.success_latched[done] = False

    # ---- helpers ----
    def _compute_reward(self, state: RenderEnvState) -> np.ndarray:
        data: SceneData = state.data
        info: Dict[str, np.ndarray] = state.info
        cube_pose = np.stack(
            [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.cube_bodies],
            axis=1,
        )
        
        ee_pos = self.robot.get_ee_pose(data)[:, :3]

        idx = np.arange(self.num_envs)
        top_pos = cube_pose[idx, self.top_idx, :3]
        base_pos = cube_pose[idx, self.base_idx, :3]

        dist_ee_obj = np.linalg.norm(ee_pos - top_pos, axis=1)
        dist_xy = np.linalg.norm(top_pos[:, :2] - base_pos[:, :2], axis=1)
        dz = top_pos[:, 2] - base_pos[:, 2]

        grip_cmd = np.asarray(data.actuator_ctrls)[:, self.robot.gripper_act_id]
        grip_closed = grip_cmd > float(self._cfg.gripper_close_thresh)

        is_grasp = (dist_ee_obj < float(self._cfg.grasp_dist_thresh)) & grip_closed
        self.grasp_latched = self.grasp_latched | is_grasp

        is_stack_pos = (dist_xy < float(self._cfg.success_dist_xy)) & (
            (dz > float(self._cfg.success_delta_z_min)) & (dz < float(self._cfg.success_delta_z_max))
        )
        is_success = is_stack_pos & (~grip_closed)
        self.success_latched = self.success_latched | is_success

        reach_r = -dist_ee_obj
        stack_r = -dist_xy
        grasp_r = 0.5 * self.grasp_latched.astype(np.float32)
        success_r = 5.0 * self.success_latched.astype(np.float32)

        reward = reach_r + stack_r + grasp_r + success_r

        # Write metrics into info in-place
        info["is_success"] = self.success_latched.copy()
        info["is_grasped"] = self.grasp_latched.copy()
        info["reach_dist"] = dist_ee_obj
        info["stack_xy"] = dist_xy
        info["dz"] = dz

        return reward.astype(np.float32)

    # def _check_success(self, state: RenderEnvState) -> np.ndarray:
    #     data: SceneData = state.data
    #     B = self.num_envs
    #     cube_pose = np.stack(
    #         [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.cube_bodies],
    #         axis=1,
    #     )
    #     name_to_idx = {name: i for i, name in enumerate(self._cfg.cube_names)}
    #     yellow_i = name_to_idx["cube_yellow"]
    #     orange_i = name_to_idx["cube_orange"]

    #     yellow = cube_pose[:, yellow_i, :3]
    #     orange = cube_pose[:, orange_i, :3]

    #     dist_xy = np.linalg.norm(yellow[:, :2] - orange[:, :2], axis=1)
    #     dz = np.abs(yellow[:, 2] - (orange[:, 2] + 0.05))

    #     success = (dist_xy < 0.01) & (dz < 0.01)
    #     # latch
    #     self.success_latched = self.success_latched | success
    #     return self.success_latched.copy()

    # def update_state(self, state: RenderEnvState, obs_required: bool = True) -> RenderEnvState:
    #     state = super().update_state(state, obs_required=obs_required)

    #     # Fail-fast check: if any cube leaves workspace bounds, terminate env
    #     data: SceneData = state.data
    #     cube_pose = np.stack(
    #         [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.cube_bodies],
    #         axis=1,
    #     )
    #     xy = cube_pose[..., :2]  # (B, num_cubes, 2)
    #     x_ok = (xy[..., 0] >= -0.75) & (xy[..., 0] <= -0.45)
    #     y_ok = (xy[..., 1] >= -0.20) & (xy[..., 1] <= 0.20)
    #     in_bounds = x_ok & y_ok
    #     out_of_bounds = ~np.all(in_bounds, axis=1)

    #     if np.any(out_of_bounds):
    #         terminated = state.terminated.copy()
    #         terminated[out_of_bounds] = True
    #         state.terminated = terminated
    #         state.info["out_of_bounds"] = out_of_bounds

    #     return state