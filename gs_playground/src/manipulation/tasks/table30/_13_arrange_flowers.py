from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np
from motrixsim import SceneData
from scipy.spatial.transform import Rotation  # noqa: F401 (kept for compatibility)
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
    "vase": _ASSETS_TASK_DIR / "3dgs" / "vase.ply",
    "vase2": _ASSETS_TASK_DIR / "3dgs" / "vase2.ply",
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
            / "test.xml"
        ).as_posix()
    )

    # control
    action_mode: str = "eef_relative"

    # rendering
    img_width: int = 320
    img_height: int = 240
    cam_id: Sequence[int] = field(default_factory=lambda: [0, 1])

    # observation / prompt
    instruction: str = (
        "Take the flower out of the transparent vase and put it into the green vase, then take it out of the green vase and put it back into the transparent vase."
    )

    # task entities
    flower_name: str = "flower"
    vase_name: str = "vase"       # source vase
    vase2_name: str = "vase2"     # target vase

    # success checks
    success_dist_xy: float = 0.15
    success_z_depth: float = 0.02
    safe_z: float = -0.20

    gripper_close_thresh: float = 0.2
    grasp_dist_thresh: float = 0.05

    vase_rim_height: float = 0.35

    alignment_thresh: float = 0.45

    # -------------------------
    # Randomization (UPDATED)
    # -------------------------
    vase_min_xy_dist: float = 0.18
    vase_range_center_xy: tuple[float, float] = (0.45, -0.0)
    vase_range_half_size_xy: tuple[float, float] = (0.03, 0.15)

    ee_home_pos: tuple[float, float, float] = (0.426, 0.09, 0.55)
    ee_home_dist_thresh: float = 0.10

    # Replay-only pose lift used by older restore pipelines.
    # Collectors should normally keep this at 0.0.
    replay_z_offset: float = 0.85


@env("table30/arrange_flowers_franka", "np")
class ArrangeFlowersEnv(TaskEnv):
    """
    Task: Move flower vase -> vase2 -> vase.
    Randomization: only vase / vase2 XY, with min distance constraint.
    Flower follows source vase XY by the SAME delta (offset), so flower starts inside source vase.
    """

    def __init__(self, cfg: ArrangeFlowersEnvCfg, num_envs: int = 32):
        super().__init__(cfg, num_envs=num_envs)

        self.flower_body = self.model.get_body(self.model.get_body_index(cfg.flower_name))
        self.vase_src_body = self.model.get_body(self.model.get_body_index(cfg.vase_name))
        self.vase_dst_body = self.model.get_body(self.model.get_body_index(cfg.vase2_name))

        # Backward compatibility (if some old code expects env.vase_body)
        self.vase_body = self.vase_src_body

        # State trackers
        self.grasp_latched = np.zeros((self.num_envs,), dtype=bool)
        self.inserted_latched = np.zeros((self.num_envs,), dtype=bool)
        self.success_latched = np.zeros((self.num_envs,), dtype=bool)

        # NEW: stage tracker for two-place task
        # 0: not placed to dst yet
        # 1: placed to dst
        # 2: placed back to src => success (now also requires contact)
        self.place_stage = np.zeros((self.num_envs,), dtype=np.int32)

        # --- NEW: replay override storage ---
        # map: env_id -> dict with keys: flower_pose_wxyz, vase_src_pose_wxyz, vase_dst_pose_wxyz (each shape (7,))
        self._replay_init: dict[int, dict[str, np.ndarray]] = {}

    # -------------------------------------------------------------------------
    # Replay APIs
    # -------------------------------------------------------------------------
    def set_replay_init(
        self,
        env_id: int,
        *,
        flower_pose_wxyz: np.ndarray,
        vase_src_pose_wxyz: np.ndarray,
        vase_dst_pose_wxyz: np.ndarray,
    ) -> None:
        self._replay_init[int(env_id)] = {
            "flower_pose_wxyz": np.asarray(flower_pose_wxyz, dtype=np.float32).reshape(7),
            "vase_src_pose_wxyz": np.asarray(vase_src_pose_wxyz, dtype=np.float32).reshape(7),
            "vase_dst_pose_wxyz": np.asarray(vase_dst_pose_wxyz, dtype=np.float32).reshape(7),
        }

    def clear_replay_init(self) -> None:
        self._replay_init = {}

    # ---- Task hooks ----
    def task_gaussians(self) -> Dict[str, str]:
        return TASK_GAUSSIANS

    def _reset_task_state(self, done: np.ndarray):
        """Reset latches & stage."""
        self.grasp_latched[done] = False
        self.inserted_latched[done] = False
        self.success_latched[done] = False
        self.place_stage[done] = 0

    def _check_flower_alignment(self, flower_quat_wxyz: np.ndarray) -> np.ndarray:
        """
        Alignment score: -Y(local) aligned with +Z(world).
        flower_quat_wxyz: (B,4) wxyz
        """
        w, x, y, z = (
            flower_quat_wxyz[:, 0],
            flower_quat_wxyz[:, 1],
            flower_quat_wxyz[:, 2],
            flower_quat_wxyz[:, 3],
        )
        vec_y_z_comp = 2.0 * (y * z + w * x)
        alignment_score = -vec_y_z_comp
        return alignment_score

    def _randomize(self, data: SceneData, done_mask: np.ndarray, phase: str = "reset"):
        """
        Only randomize source/target vase XY with min distance.
        Then move flower XY by the SAME delta as source vase moved.

        Replay override:
          If all resetting envs have replay init poses provided via set_replay_init(),
          we will directly write those poses and skip randomization.
        """
        env_ids = np.where(done_mask)[0]
        n_reset = int(env_ids.size)
        if n_reset == 0:
            return

        # -------------------------------------------------------------------------
        # Replay override: if all resetting envs have provided initial poses, use them
        # -------------------------------------------------------------------------
        if self._replay_init:
            if all(int(eid) in self._replay_init for eid in env_ids.tolist()):
                new_vase_src = np.stack(
                    [self._replay_init[int(eid)]["vase_src_pose_wxyz"] for eid in env_ids.tolist()],
                    axis=0,
                ).astype(np.float32)
                new_vase_dst = np.stack(
                    [self._replay_init[int(eid)]["vase_dst_pose_wxyz"] for eid in env_ids.tolist()],
                    axis=0,
                ).astype(np.float32)
                new_flower = np.stack(
                    [self._replay_init[int(eid)]["flower_pose_wxyz"] for eid in env_ids.tolist()],
                    axis=0,
                ).astype(np.float32)
                z_offset = float(getattr(self._cfg, "replay_z_offset", 0.0))
                if z_offset != 0.0:
                    new_vase_src[:, 2] += z_offset
                    new_vase_dst[:, 2] += z_offset
                    new_flower[:, 2] += z_offset

                self.vase_src_body.mocap.set_pose(data, new_vase_src)
                self.vase_dst_body.mocap.set_pose(data, new_vase_dst)
                self.flower_body.set_dof_pos(data, new_flower, include_floatingbase=True)
                return

        # -------------------------------------------------------------------------
        # Hard-coded base poses (shape must be (n_reset, 7) for batched writeback)
        # pose format assumed: [x, y, z, qw, qx, qy, qz] (wxyz)
        # -------------------------------------------------------------------------
        base_vase_src_pose = np.array(
            [0.500104, 0.155969, 0.128147, -0.05372239, -0.0272153, -0.00775212, 0.9981549],
            dtype=np.float32,
        )
        base_vase_dst_pose = np.array(
            [0.5, 0.0, 0.125, -0.0, -0.0, -0.0, 1.0],
            dtype=np.float32,
        )
        vase_src_pose = np.repeat(base_vase_src_pose[None, :], n_reset, axis=0)  # (n_reset, 7)
        vase_dst_pose = np.repeat(base_vase_dst_pose[None, :], n_reset, axis=0)  # (n_reset, 7)
        # Keep the runtime flower orientation, only restore it to the low-height collection layer.
        flower_pose = np.asarray(self.flower_body.get_pose(data), dtype=np.float32).reshape(n_reset, -1)
        flower_pose[:, 2] = 0.3088754

        new_vase_src = vase_src_pose.copy()
        new_vase_dst = vase_dst_pose.copy()
        new_flower = flower_pose.copy()

        # -------------------------------------------------------------------------
        # Sampling range and constraint
        # -------------------------------------------------------------------------
        center = np.array(self._cfg.vase_range_center_xy, dtype=np.float32)       # (2,)
        half = np.array(self._cfg.vase_range_half_size_xy, dtype=np.float32)     # (2,)
        lower = center - half
        upper = center + half

        min_dist = float(self._cfg.vase_min_xy_dist)
        max_tries = 200

        # Sample per-env vase positions with distance constraint
        cand_src_xy = np.zeros((n_reset, 2), dtype=np.float32)
        cand_dst_xy = np.zeros((n_reset, 2), dtype=np.float32)

        remaining = np.ones((n_reset,), dtype=bool)
        for _ in range(max_tries):
            if not remaining.any():
                break

            idx = np.where(remaining)[0]
            m = int(idx.size)

            src_xy = self._rng.uniform(lower, upper, size=(m, 2)).astype(np.float32)
            dst_xy = self._rng.uniform(lower, upper, size=(m, 2)).astype(np.float32)

            ok = np.linalg.norm(src_xy - dst_xy, axis=1) >= min_dist
            if ok.any():
                good = idx[ok]
                cand_src_xy[good] = src_xy[ok]
                cand_dst_xy[good] = dst_xy[ok]
                remaining[good] = False

        # Fallback: use hard-coded base XY for remaining envs
        if remaining.any():
            rem = np.where(remaining)[0]
            cand_src_xy[rem] = vase_src_pose[rem, :2]
            cand_dst_xy[rem] = vase_dst_pose[rem, :2]

        # -------------------------------------------------------------------------
        # Apply: move src/dst vase XY; move flower XY by same delta as src vase
        # -------------------------------------------------------------------------
        old_src_xy = vase_src_pose[:, :2]           # (n_reset,2) from hard-coded base
        delta_src = cand_src_xy - old_src_xy        # (n_reset,2)

        new_vase_src[:, 0] = cand_src_xy[:, 0]
        new_vase_src[:, 1] = cand_src_xy[:, 1]

        new_vase_dst[:, 0] = cand_dst_xy[:, 0]
        new_vase_dst[:, 1] = cand_dst_xy[:, 1]

        new_flower[:, 0] = flower_pose[:, 0] + delta_src[:, 0]
        new_flower[:, 1] = flower_pose[:, 1] + delta_src[:, 1]

        # write back (batched, matches sliced data batch size)
        self.vase_src_body.mocap.set_pose(data, new_vase_src)
        self.vase_dst_body.mocap.set_pose(data, new_vase_dst)
        self.flower_body.set_dof_pos(data, new_flower, include_floatingbase=True)

    def _compute_reward(self, state: RenderEnvState) -> np.ndarray:
        data: SceneData = state.data
        info: Dict[str, np.ndarray] = state.info

        ee_pose = self.robot.get_ee_pose(data)
        ee_pos = ee_pose[:, :3]

        grip_cmd = np.asarray(data.actuator_ctrls)[:, self.robot.gripper_act_id]
        grip_closed = grip_cmd > float(self._cfg.gripper_close_thresh)
        grip_open = ~grip_closed

        flower_pose = np.asarray(self.flower_body.get_pose(data), dtype=np.float32)
        flower_pos = flower_pose[:, :3]
        flower_quat = flower_pose[:, 3:]  # wxyz

        vase_src_pos = np.asarray(self.vase_src_body.get_pose(data), dtype=np.float32)[:, :3]
        vase_dst_pos = np.asarray(self.vase_dst_body.get_pose(data), dtype=np.float32)[:, :3]

        vase_src_rim = vase_src_pos.copy()
        vase_src_rim[:, 2] = float(self._cfg.vase_rim_height)
        vase_dst_rim = vase_dst_pos.copy()
        vase_dst_rim[:, 2] = float(self._cfg.vase_rim_height)

        dist_ee_flower = np.linalg.norm(ee_pos - flower_pos, axis=1)

        dist_xy_src = np.linalg.norm(flower_pos[:, :2] - vase_src_rim[:, :2], axis=1)
        dist_xy_dst = np.linalg.norm(flower_pos[:, :2] - vase_dst_rim[:, :2], axis=1)

        dz_src = flower_pos[:, 2] - vase_src_rim[:, 2]
        dz_dst = flower_pos[:, 2] - vase_dst_rim[:, 2]

        align_score = self._check_flower_alignment(flower_quat)
        is_aligned_pose = align_score < (-1) * float(self._cfg.alignment_thresh)

        # Grasp latch (optional)
        is_grasp_dist = dist_ee_flower < float(self._cfg.grasp_dist_thresh)
        is_grasped = is_grasp_dist & grip_closed
        self.grasp_latched = self.grasp_latched | is_grasped

        # In-vase checks
        is_xy_near_src = dist_xy_src < float(self._cfg.success_dist_xy)
        is_xy_near_dst = dist_xy_dst < float(self._cfg.success_dist_xy)

        is_deep_src = (dz_src < float(self._cfg.success_z_depth)) & (dz_src > float(self._cfg.safe_z))
        is_deep_dst = (dz_dst < float(self._cfg.success_z_depth)) & (dz_dst > float(self._cfg.safe_z))

        in_src = is_xy_near_src & is_deep_src & is_aligned_pose
        in_dst = is_xy_near_dst & is_deep_dst & is_aligned_pose

        ee_far_src = np.linalg.norm(ee_pos - vase_src_pos, axis=1) > 0.2
        ee_far_dst = np.linalg.norm(ee_pos - vase_dst_pos, axis=1) > 0.2
        is_retracted1 = (ee_far_src & ee_far_dst) | (ee_pos[:, 2] > 0.4)

        # Retracted (Z-only): EE close to home height
        home_z = float(self._cfg.ee_home_pos[2])
        ee_home_dz = np.abs(ee_pos[:, 2] - home_z)
        is_retracted2 = ee_home_dz < float(self._cfg.ee_home_dist_thresh)

        # Stage updates: only count "placed" when released AND retracted
        st = self.place_stage

        placed_to_dst = (st == 0) & in_dst & grip_open & is_retracted1
        st = np.where(placed_to_dst, 1, st)

        placed_back_src = (st == 1) & in_src & grip_open & is_retracted2
        st = np.where(placed_back_src, 2, st)

        self.place_stage[:] = st

        # In this restored replay-oriented variant, do not require a contact sensor
        # because current MotrixSim builds reject touch sensors attached to mocap links.
        found = np.zeros((self.num_envs,), dtype=np.float32)
        flower_vase_in_contact = np.zeros((self.num_envs,), dtype=bool)
        instant_success = self.place_stage >= 2
        self.success_latched = self.success_latched | instant_success

        # Reward: simple shaped (for debug), not critical for collector
        reward = np.zeros(self.num_envs, dtype=np.float32)
        reward += 1.0 * (1.0 / (1.0 + dist_ee_flower))
        reward += 2.0 * self.grasp_latched.astype(np.float32)
        reward += 3.0 * (self.place_stage >= 1).astype(np.float32)
        reward += 5.0 * (self.place_stage >= 2).astype(np.float32)

        # Info
        info["is_success"] = self.success_latched.copy()
        info["is_success_instant"] = instant_success.copy()
        info["place_stage"] = self.place_stage.copy()
        info["align_score"] = align_score.copy()
        info["flower_vase_contact_found"] = found.copy()
        info["flower_vase_in_contact"] = flower_vase_in_contact.astype(np.float32)

        return reward.astype(np.float32)

    def _check_success(self, state: RenderEnvState) -> np.ndarray:
        return self.success_latched.copy()
