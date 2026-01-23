from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from motrixsim import SceneData

from gs_playground import ROOT_PATH
from gs_playground.src.env.registry import envcfg, env
from gs_playground.src.manipulation.tasks.task_env import TaskEnvCfg, TaskEnv
from gs_playground.src.env.motrix_env.render_env import RenderEnvState

# -----------------------------------------------------------------------------
# Asset Paths
# -----------------------------------------------------------------------------
_ASSETS_TASK_DIR = ROOT_PATH / "models" / "tasks" / "table30" / "_13_arrange_flowers"

TASK_GAUSSIANS = {
    "flower": _ASSETS_TASK_DIR / "3dgs" / "flower1.ply",
    "vase": _ASSETS_TASK_DIR / "3dgs" / "transparent_vase.ply",
}


@envcfg("table30/arrange_flowers_franka")
@dataclass
class ArrangeFlowersEnvCfg(TaskEnvCfg):
    # model / sim
    robot_name: str = "franka_robotiq"
    model_file: str = str((ROOT_PATH / "models" / "robots" / 
                           "manipulation" / "franka_emika_panda_robotiq" / 
                           "xmls" / "table30_13_arrange_flower.xml").as_posix())

    # control
    action_mode: str = "eef_relative" 

    # rendering
    img_width: int = 320
    img_height: int = 240
    cam_id: int = 0

    # observation / prompt
    instruction: str = "Pick up the flower, orient its -Y axis upwards, and insert it into the vase."

    # task entities
    flower_name: str = "flower"
    vase_name: str = "vase"
    
    # task params
    success_dist_xy: float = 0.10
    success_z_depth: float = 0.02  # 插入深度阈值
    safe_z = -0.20
    
    gripper_close_thresh: float = 0.2
    grasp_dist_thresh: float = 0.05
    
    vase_rim_height: float = 0.35


    # Sensors (需要在 XML 定义)
    touch_name_flower: str = "flower_touch" 
    
    # Alignment Threshold (cos theta > 0.9 is approx < 25 degrees error)
    alignment_thresh: float = 0.9


@env("table30/arrange_flowers_franka", "np")
class ArrangeFlowersEnv(TaskEnv):
    """
    Task: Arrange Flowers.
    Target Alignment: Flower's local -Y axis should verify with World Z axis.
    Randomization: Within geom 'range2' box, no rotation noise.
    """

    def __init__(self, cfg: ArrangeFlowersEnvCfg, num_envs: int = 32):
        super().__init__(cfg, num_envs=num_envs)

        self.flower_body = self.model.get_body(self.model.get_body_index(cfg.flower_name))
        self.vase_body = self.model.get_body(self.model.get_body_index(cfg.vase_name))
        
        # State trackers
        self.grasp_latched = np.zeros((self.num_envs,), dtype=bool)
        self.inserted_latched = np.zeros((self.num_envs,), dtype=bool)
        self.success_latched = np.zeros((self.num_envs,), dtype=bool)

    # ---- Task hooks ----
    def task_gaussians(self) -> Dict[str, str]:
        return TASK_GAUSSIANS

    def _randomize(self, data: SceneData, done_mask: np.ndarray, phase: str = "reset"):
        """
        修正后的随机化逻辑
        """
        # 这里的 data 已经是被 done_mask 过滤过的切片数据了
        n_reset = data.shape[0] 
        if n_reset == 0:
            return

        # 1. Position Randomization
        center_pos = np.array([0.45, -0.06], dtype=np.float32)
        half_size = np.array([0.1, 0.1], dtype=np.float32)
        
        # 直接生成对应数量的随机位置
        rand_xy_offset = self._rng.uniform(
            -half_size, 
            half_size, 
            size=(n_reset, 2)
        ).astype(np.float32)
        
        target_xy = center_pos + rand_xy_offset
        
        # 2. 获取当前 pose (维度为 n_reset, 7)
        current_pose = np.asarray(self.flower_body.get_pose(data), dtype=np.float32)
        
        # 创建新的 pose 数组 (n_reset, 7)
        new_flower_poses = current_pose.copy()
        
        # --- 关键修正：直接赋值，不需要再用 done_mask 索引 ---
        new_flower_poses[:, 0] = target_xy[:, 0]
        new_flower_poses[:, 1] = target_xy[:, 1]
        
        # 固定 Z 高度
        new_flower_poses[:, 2] = 0.08 

        # 3. 应用到仿真
        # 这里传入的 data 是切片，set_dof_pos 会正确对应到这些切片环境
        self.flower_body.set_dof_pos(
            data,
            new_flower_poses,
            include_floatingbase=True,
        )
    def _reset_task_state(self, done: np.ndarray):
        """Reset latches."""
        self.grasp_latched[done] = False
        self.inserted_latched[done] = False
        self.success_latched[done] = False

    def _check_flower_alignment(self, flower_quat: np.ndarray) -> np.ndarray:
        """
        Check if flower's negative Y axis is aligned with World Z axis.
        
        Math:
        Let q = [w, x, y, z] (scalar first).
        The local Y axis (0, 1, 0) expressed in World frame is the 2nd column of Rotation Matrix.
        R_col1 = [ 2(xy - wz), 1 - 2(x^2 + z^2), 2(yz + wx) ]^T
        
        We want local -Y to align with World Z (0, 0, 1).
        This means the Z-component of local Y should be -1.
        Z-comp of Y = 2 * (y * z + w * x)
        
        Target: 2 * (y * z + w * x) approx -1.
        Or: -1 * (2 * (y * z + w * x)) approx 1.
        """
        w, x, y, z = flower_quat[:, 0], flower_quat[:, 1], flower_quat[:, 2], flower_quat[:, 3]
        
        # Calculate Z component of the local Y axis
        # Note: Depending on quaternion convention, check if w is first. MotrixSim usually w first.
        vec_y_z_comp = 2.0 * (y * z + w * x)
        
        # We want -Y axis to point up, so we want vec_y_z_comp to be -1.
        # Score = Dot(Local_Neg_Y, World_Z) = -1 * vec_y_z_comp
        alignment_score = -vec_y_z_comp
        
        return alignment_score # Range [-1, 1], 1 is perfect alignment

    def _compute_reward(self, state: RenderEnvState) -> np.ndarray:
        data: SceneData = state.data
        info: Dict[str, np.ndarray] = state.info
        
        # 1. Robot State
        ee_pos = self.robot.get_ee_pose(data)[:, :3]
        grip_cmd = np.asarray(data.actuator_ctrls)[:, self.robot.gripper_act_id]
        grip_closed = grip_cmd > float(self._cfg.gripper_close_thresh)

        # 2. Object States
        flower_pose = np.asarray(self.flower_body.get_pose(data), dtype=np.float32)
        flower_pos = flower_pose[:, :3]
        flower_quat = flower_pose[:, 3:]
        
        vase_pos = np.asarray(self.vase_body.get_pose(data), dtype=np.float32)[:, :3]
        vase_rim_pos = vase_pos.copy()
        vase_rim_pos[:, 2] = self._cfg.vase_rim_height

        # 3. Distances & Alignment
        dist_ee_flower = np.linalg.norm(ee_pos - flower_pos, axis=1)
        dist_xy_flower_vase = np.linalg.norm(flower_pos[:, :2] - vase_rim_pos[:, :2], axis=1)
        # 垂直距离: < 0 表示在瓶口平面以下
        dist_z_flower_vase = flower_pos[:, 2] - vase_rim_pos[:, 2] 
        
        # 计算对齐度 (-Y vs World Z)
        align_score = self._check_flower_alignment(flower_quat)
        is_aligned_pose = abs(align_score) > self._cfg.alignment_thresh

        # 4. Status Checks
        # A. Grasp
        is_grasp_dist = dist_ee_flower < self._cfg.grasp_dist_thresh
        is_grasped = is_grasp_dist & grip_closed
        self.grasp_latched = self.grasp_latched | is_grasped

        # B. Insert Logic
        # 必须同时满足: XY对齐 + 姿态正确 + Z深度足够
        is_xy_near = dist_xy_flower_vase < self._cfg.success_dist_xy
        is_deep_enough = (dist_z_flower_vase < -self._cfg.success_z_depth) & (dist_z_flower_vase > self._cfg.safe_z)
        # print("flower_pose",flower_pose)
        # print("vase_pos",vase_pos)
        # print("dist_xy_flower_vase",dist_xy_flower_vase)
        # print("dist_z_flower_vase",dist_z_flower_vase)
        # print("align_score",align_score)
        is_inserted = is_xy_near & is_deep_enough & is_aligned_pose
        self.inserted_latched = self.inserted_latched | is_inserted

        # C. Success (Inserted + Released + Retracted)
        is_released = ~grip_closed
        # 简单的归位判断：手远离瓶口 或者 手高度很高
        is_retracted = (np.linalg.norm(ee_pos - vase_pos, axis=1) > 0.2) | (ee_pos[:, 2] > 0.4)
        # print("self.inserted_latched ",self.inserted_latched )
        # print("is_retracted",is_retracted)
        is_success = self.inserted_latched & is_released & is_retracted
        self.success_latched = self.success_latched | is_success

        # 5. Reward Calculation
        reward = np.zeros(self.num_envs, dtype=np.float32)

        # Stage 1: Reach
        reward += 1.0 * (1.0 / (1.0 + dist_ee_flower))
        
        # Stage 2: Grasp
        reward += 2.0 * self.grasp_latched.astype(np.float32)
        
        # Stage 3: Align & Move to Vase Rim
        # 如果已经抓住，鼓励调整姿态并移动到瓶口上方
        manipulating_mask = self.grasp_latched & (~self.inserted_latched)
        if manipulating_mask.any():
            # Alignment reward: map [-1, 1] to [0, 1] roughly, focusing on positive side
            # align_score 越接近 1 越好
            reward[manipulating_mask] += 1.5 * np.clip(align_score[manipulating_mask], 0, 1)
            # Approach Vase XY reward
            reward[manipulating_mask] += 2.0 * (1.0 / (1.0 + dist_xy_flower_vase[manipulating_mask]))

        # Stage 4: Insert (Push Down)
        # 如果已经在瓶口附近且姿态对齐，鼓励 Z 轴向下
        ready_to_insert = manipulating_mask & is_xy_near & is_aligned_pose
        if ready_to_insert.any():
            # 目标是让 dist_z 变为负数 (进入瓶子)
            # 使用 tanh 鼓励 dist_z 接近 -success_z_depth
            z_err = np.abs(dist_z_flower_vase[ready_to_insert] + self._cfg.success_z_depth)
            reward[ready_to_insert] += 3.0 * (1.0 / (1.0 + 5.0 * z_err))

        # Stage 5: Completion
        reward += 5.0 * self.inserted_latched.astype(np.float32)
        reward += 10.0 * self.success_latched.astype(np.float32)

        # 6. Info
        info["is_success"] = self.success_latched.copy()
        info["is_grasped"] = self.grasp_latched.copy()
        info["is_inserted"] = self.inserted_latched.copy()
        info["align_score"] = align_score
        
        return reward.astype(np.float32)