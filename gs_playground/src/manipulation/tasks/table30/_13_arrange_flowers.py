# =============================================================================
# File: gs_playground/src/manipulation/tasks/table30/_13_arrange_flowers.py
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from motrixsim import SceneData
from scipy.spatial.transform import Rotation

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
    model_file: str = str(
        (
            ROOT_PATH
            / "models"
            / "robots"
            / "manipulation"
            / "franka_emika_panda_robotiq"
            / "xmls"
            / "table30_13_arrange_flower.xml"
        ).as_posix()
    )

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
    alignment_thresh: float = 0.6

    # Randomization: flower yaw about WORLD Z
    rand_yaw_deg_min: float = -45.0
    rand_yaw_deg_max: float = 45.0


@env("table30/arrange_flowers_franka", "np")
class ArrangeFlowersEnv(TaskEnv):
    """
    Task: Arrange Flowers.
    Target Alignment: Flower's local -Y axis should verify with World Z axis.
    Randomization: Within geom 'range2' box, with yaw noise around world-Z.
    """

    def __init__(self, cfg: ArrangeFlowersEnvCfg, num_envs: int = 32):
        super().__init__(cfg, num_envs=num_envs)

        self.flower_body = self.model.get_body(self.model.get_body_index(cfg.flower_name))
        self.vase_body = self.model.get_body(self.model.get_body_index(cfg.vase_name))

        # State trackers
        self.grasp_latched = np.zeros((self.num_envs,), dtype=bool)
        self.inserted_latched = np.zeros((self.num_envs,), dtype=bool)
        self.success_latched = np.zeros((self.num_envs,), dtype=bool)

        # Random yaw caches (full env size; indexed by global env id)
        self.rand_yaw_rad = np.zeros((self.num_envs,), dtype=np.float32)
        self.rand_yaw_deg = np.zeros((self.num_envs,), dtype=np.float32)

    # ---- Task hooks ----
    def task_gaussians(self) -> Dict[str, str]:
        return TASK_GAUSSIANS

    @staticmethod
    def _wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
        # q_wxyz: (..., 4) -> (..., 4) xyzw
        out = np.empty_like(q_wxyz)
        out[..., 0] = q_wxyz[..., 1]
        out[..., 1] = q_wxyz[..., 2]
        out[..., 2] = q_wxyz[..., 3]
        out[..., 3] = q_wxyz[..., 0]
        return out

    @staticmethod
    def _xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
        out = np.empty_like(q_xyzw)
        out[..., 0] = q_xyzw[..., 3]
        out[..., 1] = q_xyzw[..., 0]
        out[..., 2] = q_xyzw[..., 1]
        out[..., 3] = q_xyzw[..., 2]
        return out

    def _randomize(self, data: SceneData, done_mask: np.ndarray, phase: str = "reset"):
        """
        Randomize flower position and yaw. Yaw is about WORLD Z axis, uniform in [-45, 45] deg.
        Note:
          - 'data' is typically a sliced SceneData containing only done envs.
          - done_mask is full-size (num_envs,) mask; use it to map back to global env indices.
        """
        env_ids = np.where(done_mask)[0]
        n_reset = int(env_ids.size)
        if n_reset == 0:
            return

        # 1) Position Randomization (XY in a box)
        center_pos = np.array([0.45, -0.1], dtype=np.float32)
        half_size = np.array([0.05, 0.05], dtype=np.float32)

        rand_xy_offset = self._rng.uniform(-half_size, half_size, size=(n_reset, 2)).astype(np.float32)
        target_xy = center_pos + rand_xy_offset

        # 2) Current pose in sliced data: (n_reset, 7) = [x,y,z, qw,qx,qy,qz] (wxyz)
        current_pose = np.asarray(self.flower_body.get_pose(data), dtype=np.float32)
        new_flower_poses = current_pose.copy()

        new_flower_poses[:, 0] = target_xy[:, 0]
        new_flower_poses[:, 1] = target_xy[:, 1]
        new_flower_poses[:, 2] = 0.08  # fixed Z

        # 3) Yaw randomization about WORLD Z
        yaw_deg = self._rng.uniform(
            float(self._cfg.rand_yaw_deg_min),
            float(self._cfg.rand_yaw_deg_max),
            size=(n_reset,),
        ).astype(np.float32)
        yaw_rad = (yaw_deg * np.pi / 180.0).astype(np.float32)

        # cache to full env arrays
        self.rand_yaw_deg[env_ids] = yaw_deg
        self.rand_yaw_rad[env_ids] = yaw_rad

        # Apply yaw rotation to object quaternion using scipy Rotation
        # Object quat in pose is assumed wxyz; scipy uses xyzw.
        q_cur_wxyz = new_flower_poses[:, 3:7]
        q_cur_xyzw = self._wxyz_to_xyzw(q_cur_wxyz)

        r_cur = Rotation.from_quat(q_cur_xyzw)                  # (n_reset,)
        r_yaw = Rotation.from_euler("z", yaw_rad, degrees=False)  # (n_reset,)
        r_new = r_yaw * r_cur  # left-mul: world/extrinsic Z yaw
        q_new_xyzw = r_new.as_quat().astype(np.float32)
        q_new_wxyz = self._xyzw_to_wxyz(q_new_xyzw)

        new_flower_poses[:, 3:7] = q_new_wxyz

        # 4) Apply to sim (sliced data)
        self.flower_body.set_dof_pos(
            data,
            new_flower_poses,
            include_floatingbase=True,
        )

        # 5) Best-effort: write into state.info immediately (so reset() can expose it)
        try:
            if hasattr(self, "_state") and hasattr(self._state, "info") and isinstance(self._state.info, dict):
                if "rand_yaw_rad" not in self._state.info:
                    self._state.info["rand_yaw_rad"] = np.zeros((self.num_envs,), dtype=np.float32)
                if "rand_yaw_deg" not in self._state.info:
                    self._state.info["rand_yaw_deg"] = np.zeros((self.num_envs,), dtype=np.float32)
                self._state.info["rand_yaw_rad"][env_ids] = yaw_rad
                self._state.info["rand_yaw_deg"][env_ids] = yaw_deg
        except Exception:
            pass

    def _reset_task_state(self, done: np.ndarray):
        """Reset latches."""
        self.grasp_latched[done] = False
        self.inserted_latched[done] = False
        self.success_latched[done] = False

    def _check_flower_alignment(self, flower_quat: np.ndarray) -> np.ndarray:
        """
        Check if flower's negative Y axis is aligned with World Z axis.

        Here flower_quat is assumed in wxyz (scalar first), consistent with MotrixSim usage in this project.
        """
        w, x, y, z = flower_quat[:, 0], flower_quat[:, 1], flower_quat[:, 2], flower_quat[:, 3]

        # Z component of the local Y axis: 2*(y*z + w*x)
        vec_y_z_comp = 2.0 * (y * z + w * x)

        # Want -Y pointing up => alignment score = dot(-Y_world, Z_world) = -vec_y_z_comp
        alignment_score = -vec_y_z_comp
        return alignment_score  # [-1, 1], 1 is perfect

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

        # vertical distance: <0 means below rim plane
        dist_z_flower_vase = flower_pos[:, 2] - vase_rim_pos[:, 2]

        align_score = self._check_flower_alignment(flower_quat)
        is_aligned_pose = np.abs(align_score) > self._cfg.alignment_thresh

        # 4. Status Checks
        # A. Grasp
        is_grasp_dist = dist_ee_flower < self._cfg.grasp_dist_thresh
        is_grasped = is_grasp_dist & grip_closed
        self.grasp_latched = self.grasp_latched | is_grasped

        # B. Insert Logic
        is_xy_near = dist_xy_flower_vase < self._cfg.success_dist_xy
        is_deep_enough = (dist_z_flower_vase < self._cfg.success_z_depth) & (dist_z_flower_vase > self._cfg.safe_z)
        is_inserted = is_xy_near & is_deep_enough & is_aligned_pose
        self.inserted_latched = self.inserted_latched | is_inserted

        # C. Success
        is_released = ~grip_closed
        is_retracted = (np.linalg.norm(ee_pos - vase_pos, axis=1) > 0.2) | (ee_pos[:, 2] > 0.4)
        is_success = self.inserted_latched & is_released & is_retracted
        self.success_latched = self.success_latched | is_success
        print("is_success",self.success_latched)
        print("flower_pose",flower_pose)
        print("vase_pos",vase_pos)
        print("dist_xy_flower_vase",dist_xy_flower_vase)
        print("dist_z_flower_vase",dist_z_flower_vase)
        print("align_score",align_score)
        # 5. Reward Calculation
        reward = np.zeros(self.num_envs, dtype=np.float32)

        # Stage 1: Reach
        reward += 1.0 * (1.0 / (1.0 + dist_ee_flower))

        # Stage 2: Grasp
        reward += 2.0 * self.grasp_latched.astype(np.float32)

        # Stage 3: Align & Move to Vase Rim
        manipulating_mask = self.grasp_latched & (~self.inserted_latched)
        if manipulating_mask.any():
            reward[manipulating_mask] += 1.5 * np.clip(align_score[manipulating_mask], 0, 1)
            reward[manipulating_mask] += 2.0 * (1.0 / (1.0 + dist_xy_flower_vase[manipulating_mask]))

        # Stage 4: Insert
        ready_to_insert = manipulating_mask & is_xy_near & is_aligned_pose
        if ready_to_insert.any():
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

        # NEW: export random yaw to collector
        info["rand_yaw_deg"] = self.rand_yaw_deg.copy()
        info["rand_yaw_rad"] = self.rand_yaw_rad.copy()

        return reward.astype(np.float32)
