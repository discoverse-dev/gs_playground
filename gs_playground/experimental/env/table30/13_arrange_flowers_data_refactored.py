from __future__ import annotations

import os
import time
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from gs_playground.src.manipulation.tasks.table30._13_arrange_flowers import (
    ArrangeFlowersEnv,
    ArrangeFlowersEnvCfg,
)

from table30_collect_common import (
    VideoCfg,
    PoseFixCfg,
    Table30CollectorIO,
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def smooth_step_pos(curr: np.ndarray, tgt: np.ndarray, max_dp: Any) -> np.ndarray:
    """Smooth move with step-size cap. curr/tgt: (B,3), max_dp can be scalar or (B,1)/(B,)."""
    dp = tgt - curr
    n = np.linalg.norm(dp, axis=1, keepdims=True)
    s = np.minimum(1.0, max_dp / (n + 1e-9))
    return curr + dp * s


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-9
    return q / n


def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def closest_yaw(target: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Map target yaw to the equivalent angle (2π-periodic) that is closest to curr yaw."""
    d = wrap_to_pi(target - curr)
    return curr + d


def quat_to_yaw_xyzw(q: np.ndarray) -> np.ndarray:
    """Return yaw (Z-euler) from xyzw quaternion array (...,4)."""
    return Rotation.from_quat(q).as_euler("xyz", degrees=False)[..., 2]


def obs_ee_quat_xyzw(obs: Dict[str, Any], fallback_quat_xyzw: np.ndarray) -> np.ndarray:
    """Try to read ee quaternion from obs['ee_pose'] (6D or 7D), else fallback."""
    ee_pose = obs.get("ee_pose", None)
    if ee_pose is None:
        return fallback_quat_xyzw
    if ee_pose.shape[1] == 7:
        return ee_pose[:, 3:7].astype(np.float32)
    if ee_pose.shape[1] == 6:
        return Rotation.from_euler("xyz", ee_pose[:, 3:6], degrees=False).as_quat().astype(np.float32)
    return fallback_quat_xyzw


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CollectorCfg:
    # dataset
    data_size: int = 10
    num_envs: int = 2
    seed: int = 42
    save_dir: str = "./data/table30_arrange_flowers_dual_view"

    # env control
    max_ctrl_steps: int = 1400

    # motion position
    max_dp: float = 0.005
    pos_tol: float = 0.001

    # rotation control
    rot_gain: float = 0.6
    max_dr: float = 0.08
    yaw_tol: float = 0.03

    # keypoint offsets (world)
    grasp_offset: Tuple[float, float, float] = (-0.012, 0.0, 0.018)
    pregrasp_z_margin: float = 0.02
    lift_height_z: float = 0.06

    # pick1 coarse approach extra offset (XY)
    pick1_coarse_offset_xy: Tuple[float, float] = (0.0, 0.0)

    # pick2 grasp extra Z
    pick2_grasp_z_extra: float = 0.0

    # place offsets relative to target vase
    place_offset: Tuple[float, float, float] = (0.0, 0.0, 0.05)
    place_down_z: float = 0.015
    retreat_dx: float = 0.10

    # yaw offsets
    yaw_offset_pick: float = 0.0
    yaw_offset_place: float = 0.0

    # gripper
    gripper_open: float = 0.0
    gripper_close: float = 0.82

    # timing / dwell
    close_hold_steps: int = 15
    open_hold_steps: int = 15
    waypoint_dwell_steps: int = 20

    # sampling / render
    sample_every_steps: int = 1
    save_video: bool = True
    render_every_steps: int = 1
    video_fps: int = 30
    video_w: int = 640
    video_h: int = 480
    cam_view_key: Sequence[str] = field(default_factory=lambda: ["pixels/view_0", "pixels/view_1"])

    # text fields
    instruction: str = (
        "Pick up the flower from the source vase, place it into the target vase, "
        "then pick it again and adjust its placement."
    )


# -----------------------------------------------------------------------------
# Collector
# -----------------------------------------------------------------------------
class ArrangeFlowersCollector:
    # Phase 1: Pick1 (Manhattan + yaw)
    ST_P1_LIFT_Z = 0
    ST_P1_ALIGN_X = 1
    ST_P1_ALIGN_Y = 2
    ST_P1_ALIGN_YAW = 3
    ST_P1_DESCEND = 4
    ST_P1_CLOSE = 5

    # Phase 2: Place1 (Manhattan + yaw)
    ST_T1_LIFT_Z = 6
    ST_T1_ALIGN_X = 7
    ST_T1_ALIGN_Y = 8
    ST_T1_ALIGN_YAW = 9
    ST_T1_DESCEND = 10
    ST_T1_OPEN = 11

    # Phase 3: Pick2 (regrasp)
    ST_P2_LIFT_Z = 12
    ST_P2_ALIGN_X = 13
    ST_P2_ALIGN_Y = 14
    ST_P2_ALIGN_YAW = 15
    ST_P2_DESCEND = 16
    ST_P2_CLOSE = 17

    # Phase 4: Place2 (final placement)
    ST_T2_LIFT_Z = 18
    ST_T2_ALIGN_X = 19
    ST_T2_ALIGN_Y = 20
    ST_T2_ALIGN_YAW = 21
    ST_T2_DESCEND = 22
    ST_T2_OPEN = 23

    # End
    ST_RETREAT = 24
    ST_HOME = 25
    ST_DONE = 26

    def __init__(self, cfg: CollectorCfg, env_cfg: Optional[ArrangeFlowersEnvCfg] = None):
        self.cfg = cfg
        os.makedirs(cfg.save_dir, exist_ok=True)

        self.env_cfg = env_cfg if env_cfg is not None else ArrangeFlowersEnvCfg()
        self.env = ArrangeFlowersEnv(self.env_cfg, num_envs=int(cfg.num_envs))
        self.env.reset()

        self.B = int(cfg.num_envs)
        self.cam_keys = list(cfg.cam_view_key)

        # lifecycle
        B = self.B
        self.active = np.zeros(B, dtype=bool)
        self.done = np.zeros(B, dtype=bool)
        self.success = np.zeros(B, dtype=bool)
        self.ctrl_step = np.zeros(B, dtype=np.int32)
        self._attempt_id = np.zeros(B, dtype=np.int64)

        # fsm
        self.states = np.zeros(B, dtype=np.int32)
        self.state_enter_step = np.zeros(B, dtype=np.int32)
        self.state_reach_step = np.full(B, -1, dtype=np.int32)

        # controls
        self.exec_pos = np.zeros((B, 3), dtype=np.float32)
        self.exec_quat = np.zeros((B, 4), dtype=np.float32)  # xyzw

        # latch start pose/orientation
        self.latched_start_pos = np.zeros((B, 3), dtype=np.float32)
        self.latched_start_quat = np.zeros((B, 4), dtype=np.float32)
        self.latched_start_yaw = np.zeros((B,), dtype=np.float32)

        # shared IO
        pose_fix = PoseFixCfg(
            z_offset=0.1525,
            yaw_offset=-0.5 * np.pi,
            yaw_sign=-1.0,
            wrap_yaw=True,
        )
        video_cfg = VideoCfg(
            fps=int(cfg.video_fps),
            width=int(cfg.video_w),
            height=int(cfg.video_h),
            codec="h264",
        )
        self.io = Table30CollectorIO(
            save_dir=str(cfg.save_dir),
            num_envs=B,
            cam_keys=self.cam_keys,
            save_video=bool(cfg.save_video),
            sample_every_steps=int(cfg.sample_every_steps),
            render_every_steps=int(cfg.render_every_steps),
            video_cfg=video_cfg,
            pose_fix=pose_fix,
            dt=0.02,
        )

        self.saved_success = 0
        self.attempted = 0
        self._last_log_t = time.perf_counter()
        self._last_action = np.zeros((B, 7), dtype=np.float32)

        # bodies
        self.flower_body = self.env.flower_body
        self.vase_src_body = self.env.vase_src_body
        # Env naming differs across versions; prefer the current attribute name.
        self.vase_tgt_body = getattr(self.env, "vase_dst_body", None)
        if self.vase_tgt_body is None:
            self.vase_tgt_body = getattr(self.env, "vase_tgt_body", None)
        if self.vase_tgt_body is None:
            raise AttributeError(
                "ArrangeFlowersEnv is missing vase target body. Expected 'vase_dst_body' (new) or 'vase_tgt_body' (old)."
            )

    # -----------------------------
    # Episode init/reset
    # -----------------------------
    def start_episodes(self, env_ids: np.ndarray, seed: int) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if env_ids.size == 0:
            return

        try:
            self.env._rng = np.random.default_rng(int(seed))
        except Exception:
            pass

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

        ee_pose = self.env.robot.get_ee_pose(data)
        self.exec_pos[env_ids] = ee_pose[env_ids, :3]
        self.latched_start_pos[env_ids] = ee_pose[env_ids, :3]

        if ee_pose.shape[1] == 6:
            q = Rotation.from_euler("xyz", ee_pose[env_ids, 3:6], degrees=False).as_quat().astype(np.float32)
        else:
            q = ee_pose[env_ids, 3:7].astype(np.float32)
        q = normalize_quat_xyzw(q)
        self.exec_quat[env_ids] = q
        self.latched_start_quat[env_ids] = q.copy()
        self.latched_start_yaw[env_ids] = quat_to_yaw_xyzw(q).astype(np.float32)

        self.states[env_ids] = self.ST_P1_LIFT_Z

        for env_id in env_ids.tolist():
            self.io.reset_env(int(env_id))
            self._attempt_id[env_id] += 1
            self.attempted += 1

    # -----------------------------
    # FSM helper
    # -----------------------------
    def _enter_state(self, mask: np.ndarray, new_state: int) -> None:
        if not np.any(mask):
            return
        self.states[mask] = int(new_state)
        self.state_enter_step[mask] = self.ctrl_step[mask].copy()
        self.state_reach_step[mask] = -1

    # -----------------------------
    # Core logic (FSM + control)
    # -----------------------------
    def _step_logic(self) -> None:
        cfg = self.cfg
        B = self.B

        running = self.active & (~self.done)
        if not np.any(running):
            return

        data = self.env._state.data

        # live poses
        flower_pose = np.asarray(self.flower_body.get_pose(data), dtype=np.float32)
        flower_p = flower_pose[:, :3]
        flower_q = flower_pose[:, 3:7]
        flower_q = normalize_quat_xyzw(flower_q)

        vase_tgt_pose = np.asarray(self.vase_tgt_body.get_pose(data), dtype=np.float32)
        vase_tgt_p = vase_tgt_pose[:, :3]

        # keypoints helpers
        g_off = np.asarray(cfg.grasp_offset, dtype=np.float32).reshape(1, 3)

        def build_above(p: np.ndarray, dz: float) -> np.ndarray:
            out = p.copy()
            out[:, 2] = out[:, 2] + float(dz)
            return out

        # ---------------- pick1 keypoints ----------------
        pick1_dx, pick1_dy = cfg.pick1_coarse_offset_xy
        p1_above = build_above(flower_p, cfg.lift_height_z)

        p1_x = p1_above.copy()
        p1_x[:, 0] = flower_p[:, 0] + g_off[0, 0] + float(pick1_dx)

        p1_y = p1_x.copy()
        p1_y[:, 1] = flower_p[:, 1] + g_off[0, 1] + float(pick1_dy)

        p1_pre = p1_y.copy()
        p1_pre[:, 2] = flower_p[:, 2] + g_off[0, 2] + float(cfg.pregrasp_z_margin)

        p1_fine_y = p1_pre.copy()
        p1_fine_y[:, 1] = flower_p[:, 1] + g_off[0, 1]

        p1_fine_x = p1_fine_y.copy()
        p1_fine_x[:, 0] = flower_p[:, 0] + g_off[0, 0]

        p1_grasp = p1_fine_x.copy()
        p1_grasp[:, 2] = flower_p[:, 2] + g_off[0, 2]

        # ---------------- place1 keypoints ----------------
        place_off = np.asarray(cfg.place_offset, dtype=np.float32).reshape(1, 3)
        p_t1_above = build_above(vase_tgt_p + place_off, cfg.lift_height_z)

        p_t1_x = p_t1_above.copy()
        p_t1_x[:, 0] = vase_tgt_p[:, 0] + place_off[0, 0]

        p_t1_y = p_t1_x.copy()
        p_t1_y[:, 1] = vase_tgt_p[:, 1] + place_off[0, 1]

        p_t1_down = p_t1_y.copy()
        p_t1_down[:, 2] = vase_tgt_p[:, 2] + float(cfg.place_down_z)

        # ---------------- pick2: recompute from current flower pose ----------------
        p2_above = build_above(flower_p, cfg.lift_height_z)

        p2_x = p2_above.copy()
        p2_x[:, 0] = flower_p[:, 0] + g_off[0, 0]

        p2_y = p2_x.copy()
        p2_y[:, 1] = flower_p[:, 1] + g_off[0, 1]

        p2_pre = p2_y.copy()
        p2_pre[:, 2] = flower_p[:, 2] + g_off[0, 2] + float(cfg.pregrasp_z_margin)

        p2_fine_y = p2_pre.copy()
        p2_fine_y[:, 1] = flower_p[:, 1] + g_off[0, 1]

        p2_fine_x = p2_fine_y.copy()
        p2_fine_x[:, 0] = flower_p[:, 0] + g_off[0, 0]

        p2_grasp = p2_fine_x.copy()
        p2_grasp[:, 2] = flower_p[:, 2] + g_off[0, 2] + float(cfg.pick2_grasp_z_extra)

        # ---------------- place2 keypoints ----------------
        p_t2_above = p_t1_above.copy()
        p_t2_x = p_t1_x.copy()
        p_t2_y = p_t1_y.copy()
        p_t2_down = p_t1_down.copy()

        # retreat/home
        p_retreat = p_t2_above.copy()
        p_retreat[:, 0] -= float(cfg.retreat_dx)
        p_home = self.latched_start_pos.copy()
        p_home[:, 2] = p_home[:, 2] + float(cfg.lift_height_z)

        # ---------------- target assignment ----------------
        s = self.states
        tgt_pos = self.exec_pos.copy()
        tgt_yaw = np.full((B,), np.nan, dtype=np.float32)
        grip_cmd = np.full((B,), float(cfg.gripper_open), dtype=np.float32)

        def set_target(state_id: int, pos: np.ndarray, grip: float) -> None:
            mask = running & (s == state_id)
            if np.any(mask):
                tgt_pos[mask] = pos[mask]
                grip_cmd[mask] = float(grip)

        def set_target_yaw(state_id: int, yaw_arr: np.ndarray) -> None:
            mask = running & (s == state_id)
            if np.any(mask):
                tgt_yaw[mask] = yaw_arr[mask]

        # pick1
        set_target(self.ST_P1_LIFT_Z, p1_above, cfg.gripper_open)
        set_target(self.ST_P1_ALIGN_X, p1_x, cfg.gripper_open)
        set_target(self.ST_P1_ALIGN_Y, p1_y, cfg.gripper_open)
        set_target(self.ST_P1_ALIGN_YAW, p1_pre, cfg.gripper_open)
        set_target(self.ST_P1_DESCEND, p1_grasp, cfg.gripper_open)
        set_target(self.ST_P1_CLOSE, p1_grasp, cfg.gripper_close)

        # place1
        set_target(self.ST_T1_LIFT_Z, p_t1_above, cfg.gripper_close)
        set_target(self.ST_T1_ALIGN_X, p_t1_x, cfg.gripper_close)
        set_target(self.ST_T1_ALIGN_Y, p_t1_y, cfg.gripper_close)
        set_target(self.ST_T1_ALIGN_YAW, p_t1_y, cfg.gripper_close)
        set_target(self.ST_T1_DESCEND, p_t1_down, cfg.gripper_close)
        set_target(self.ST_T1_OPEN, p_t1_down, cfg.gripper_open)

        # pick2
        set_target(self.ST_P2_LIFT_Z, p2_above, cfg.gripper_open)
        set_target(self.ST_P2_ALIGN_X, p2_x, cfg.gripper_open)
        set_target(self.ST_P2_ALIGN_Y, p2_y, cfg.gripper_open)
        set_target(self.ST_P2_ALIGN_YAW, p2_pre, cfg.gripper_open)
        set_target(self.ST_P2_DESCEND, p2_grasp, cfg.gripper_open)
        set_target(self.ST_P2_CLOSE, p2_grasp, cfg.gripper_close)

        # place2
        set_target(self.ST_T2_LIFT_Z, p_t2_above, cfg.gripper_close)
        set_target(self.ST_T2_ALIGN_X, p_t2_x, cfg.gripper_close)
        set_target(self.ST_T2_ALIGN_Y, p_t2_y, cfg.gripper_close)
        set_target(self.ST_T2_ALIGN_YAW, p_t2_y, cfg.gripper_close)
        set_target(self.ST_T2_DESCEND, p_t2_down, cfg.gripper_close)
        set_target(self.ST_T2_OPEN, p_t2_down, cfg.gripper_open)

        # end
        set_target(self.ST_RETREAT, p_retreat, cfg.gripper_open)
        set_target(self.ST_HOME, p_home, cfg.gripper_open)

        # yaw policy (pick: align to flower; place: reset to start yaw with offsets)
        flower_yaw = quat_to_yaw_xyzw(flower_q).astype(np.float32)
        pick_yaw = wrap_to_pi(flower_yaw + float(cfg.yaw_offset_pick)).astype(np.float32)
        place_yaw = wrap_to_pi(self.latched_start_yaw + float(cfg.yaw_offset_place)).astype(np.float32)

        set_target_yaw(self.ST_P1_ALIGN_YAW, pick_yaw)
        set_target_yaw(self.ST_P2_ALIGN_YAW, pick_yaw)
        set_target_yaw(self.ST_T1_ALIGN_YAW, place_yaw)
        set_target_yaw(self.ST_T2_ALIGN_YAW, place_yaw)
        set_target_yaw(self.ST_HOME, self.latched_start_yaw)

        # ---------------- control: position ----------------
        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos, float(cfg.max_dp))

        # ---------------- control: orientation (yaw) ----------------
        ref_pose_6d = self.env.robot.ref_ee_pose
        ref_pos = ref_pose_6d[:, :3]
        ref_euler = ref_pose_6d[:, 3:6]
        ref_quat = Rotation.from_euler("xyz", ref_euler, degrees=False).as_quat().astype(np.float32)

        obs = self.env._state.obs
        ee_quat = obs_ee_quat_xyzw(obs, fallback_quat_xyzw=ref_quat)
        curr_yaw = quat_to_yaw_xyzw(ee_quat).astype(np.float32)

        want_rot = running & (~np.isnan(tgt_yaw))
        desired_quat = self.exec_quat.copy()
        if np.any(want_rot):
            target_y = closest_yaw(tgt_yaw[want_rot].astype(np.float32), curr_yaw[want_rot]).astype(np.float32)
            start_y = self.latched_start_yaw[want_rot].astype(np.float32)
            delta_y = wrap_to_pi(target_y - start_y).astype(np.float32)

            r_delta = Rotation.from_euler("z", delta_y)
            r_start = Rotation.from_quat(self.latched_start_quat[want_rot])
            desired_quat[want_rot] = (r_delta * r_start).as_quat().astype(np.float32)
            self.exec_quat[want_rot] = desired_quat[want_rot]

        rotvec_cmd = np.zeros((B, 3), dtype=np.float32)
        if np.any(want_rot):
            r_des = Rotation.from_quat(desired_quat[want_rot])
            r_ref = Rotation.from_quat(ref_quat[want_rot])
            r_err = r_des * r_ref.inv()
            rv = r_err.as_rotvec().astype(np.float32)

            mag = np.linalg.norm(rv, axis=1, keepdims=True) + 1e-9
            scale = np.minimum(1.0, float(cfg.max_dr) / mag)
            rotvec_cmd[want_rot] = rv * scale * float(cfg.rot_gain)

        # action
        action = np.zeros((B, 7), dtype=np.float32)
        action[:, :3] = (self.exec_pos - ref_pos) * 0.5
        action[:, 3:6] = rotvec_cmd
        action[:, 6] = grip_cmd

        self._last_action[:] = action
        self.env.step(action)

        # ---------------- transitions ----------------
        obs = self.env._state.obs
        ee_quat = obs_ee_quat_xyzw(obs, fallback_quat_xyzw=ref_quat)
        curr_yaw = quat_to_yaw_xyzw(ee_quat).astype(np.float32)

        def is_reached(target_p: np.ndarray) -> np.ndarray:
            return np.linalg.norm(self.exec_pos - target_p, axis=1) < float(cfg.pos_tol)

        def _check_reach_and_dwell(state_idx: int, target_p: np.ndarray) -> np.ndarray:
            in_state = running & (self.states == state_idx)
            reached = is_reached(target_p)
            just = in_state & reached & (self.state_reach_step == -1)
            if np.any(just):
                self.state_reach_step[just] = self.ctrl_step[just]
            dwell_pass = (self.ctrl_step - self.state_reach_step) >= int(cfg.waypoint_dwell_steps)
            return in_state & (self.state_reach_step != -1) & dwell_pass & reached

        def _check_yaw_and_dwell(state_idx: int, target_y: np.ndarray) -> np.ndarray:
            in_state = running & (self.states == state_idx)
            dy = wrap_to_pi(curr_yaw - target_y)
            ok = np.abs(dy) < float(cfg.yaw_tol)
            just = in_state & ok & (self.state_reach_step == -1)
            if np.any(just):
                self.state_reach_step[just] = self.ctrl_step[just]
            dwell_pass = (self.ctrl_step - self.state_reach_step) >= int(cfg.waypoint_dwell_steps)
            return in_state & (self.state_reach_step != -1) & dwell_pass & ok

        # pick1 chain
        self._enter_state(_check_reach_and_dwell(self.ST_P1_LIFT_Z, p1_above), self.ST_P1_ALIGN_X)
        self._enter_state(_check_reach_and_dwell(self.ST_P1_ALIGN_X, p1_x), self.ST_P1_ALIGN_Y)
        self._enter_state(_check_reach_and_dwell(self.ST_P1_ALIGN_Y, p1_y), self.ST_P1_ALIGN_YAW)
        self._enter_state(_check_yaw_and_dwell(self.ST_P1_ALIGN_YAW, pick_yaw), self.ST_P1_DESCEND)
        self._enter_state(_check_reach_and_dwell(self.ST_P1_DESCEND, p1_grasp), self.ST_P1_CLOSE)

        mask = running & (self.states == self.ST_P1_CLOSE)
        if np.any(mask):
            hold_done = (self.ctrl_step - self.state_enter_step) >= int(cfg.close_hold_steps)
            self._enter_state(mask & hold_done, self.ST_T1_LIFT_Z)

        # place1 chain
        self._enter_state(_check_reach_and_dwell(self.ST_T1_LIFT_Z, p_t1_above), self.ST_T1_ALIGN_X)
        self._enter_state(_check_reach_and_dwell(self.ST_T1_ALIGN_X, p_t1_x), self.ST_T1_ALIGN_Y)
        self._enter_state(_check_reach_and_dwell(self.ST_T1_ALIGN_Y, p_t1_y), self.ST_T1_ALIGN_YAW)
        self._enter_state(_check_yaw_and_dwell(self.ST_T1_ALIGN_YAW, place_yaw), self.ST_T1_DESCEND)
        self._enter_state(_check_reach_and_dwell(self.ST_T1_DESCEND, p_t1_down), self.ST_T1_OPEN)

        mask = running & (self.states == self.ST_T1_OPEN)
        if np.any(mask):
            hold_done = (self.ctrl_step - self.state_enter_step) >= int(cfg.open_hold_steps)
            self._enter_state(mask & hold_done, self.ST_P2_LIFT_Z)

        # pick2 chain
        self._enter_state(_check_reach_and_dwell(self.ST_P2_LIFT_Z, p2_above), self.ST_P2_ALIGN_X)
        self._enter_state(_check_reach_and_dwell(self.ST_P2_ALIGN_X, p2_x), self.ST_P2_ALIGN_Y)
        self._enter_state(_check_reach_and_dwell(self.ST_P2_ALIGN_Y, p2_y), self.ST_P2_ALIGN_YAW)
        self._enter_state(_check_yaw_and_dwell(self.ST_P2_ALIGN_YAW, pick_yaw), self.ST_P2_DESCEND)
        self._enter_state(_check_reach_and_dwell(self.ST_P2_DESCEND, p2_grasp), self.ST_P2_CLOSE)

        mask = running & (self.states == self.ST_P2_CLOSE)
        if np.any(mask):
            hold_done = (self.ctrl_step - self.state_enter_step) >= int(cfg.close_hold_steps)
            self._enter_state(mask & hold_done, self.ST_T2_LIFT_Z)

        # place2 chain
        self._enter_state(_check_reach_and_dwell(self.ST_T2_LIFT_Z, p_t2_above), self.ST_T2_ALIGN_X)
        self._enter_state(_check_reach_and_dwell(self.ST_T2_ALIGN_X, p_t2_x), self.ST_T2_ALIGN_Y)
        self._enter_state(_check_reach_and_dwell(self.ST_T2_ALIGN_Y, p_t2_y), self.ST_T2_ALIGN_YAW)
        self._enter_state(_check_yaw_and_dwell(self.ST_T2_ALIGN_YAW, place_yaw), self.ST_T2_DESCEND)
        self._enter_state(_check_reach_and_dwell(self.ST_T2_DESCEND, p_t2_down), self.ST_T2_OPEN)

        mask = running & (self.states == self.ST_T2_OPEN)
        if np.any(mask):
            hold_done = (self.ctrl_step - self.state_enter_step) >= int(cfg.open_hold_steps)
            self._enter_state(mask & hold_done, self.ST_RETREAT)

        self._enter_state(_check_reach_and_dwell(self.ST_RETREAT, p_retreat), self.ST_HOME)
        self._enter_state(_check_reach_and_dwell(self.ST_HOME, p_home), self.ST_DONE)

    # -----------------------------
    # Collection loop
    # -----------------------------
    def collect(self) -> None:
        cfg = self.cfg
        target = int(cfg.data_size)

        all_ids = np.arange(self.B, dtype=np.int64)
        self.start_episodes(all_ids, seed=int(cfg.seed))

        while self.saved_success < target:
            self._step_logic()

            obs = self.env._state.obs
            running = self.active & (~self.done)

            # capture
            sample_mask = running & ((self.ctrl_step % int(cfg.sample_every_steps)) == 0)
            for env_id in np.where(sample_mask)[0].tolist():
                reward = float(np.asarray(self.env._state.reward).reshape(-1)[env_id])
                self.io.capture_step(
                    env_id=int(env_id),
                    ctrl_step=int(self.ctrl_step[env_id]),
                    state=int(self.states[env_id]),
                    obs=obs,
                    last_action=self._last_action,
                    success=bool(self.success[env_id]),
                    reward=reward,
                    extra_step=None,
                )

            # video
            if cfg.save_video:
                render_mask = running & ((self.ctrl_step % int(cfg.render_every_steps)) == 0)
                for env_id in np.where(render_mask)[0].tolist():
                    self.io.maybe_write_video(
                        obs=obs,
                        env_id=int(env_id),
                        ctrl_step=int(self.ctrl_step[env_id]),
                    )

            # advance time
            self.ctrl_step[running] += 1

            # done/success check
            env_success = np.asarray(getattr(self.env, "success_latched", np.zeros((self.B,), dtype=np.bool_))).reshape(-1)
            for i in range(self.B):
                if (not running[i]) or self.done[i]:
                    continue
                fsm_done = int(self.states[i]) == int(self.ST_DONE)
                timeout = int(self.ctrl_step[i]) >= int(cfg.max_ctrl_steps)
                if fsm_done or bool(env_success[i]) or timeout:
                    self.done[i] = True
                    self.success[i] = bool(fsm_done or env_success[i])

            # finalize & restart
            for i in range(self.B):
                if not (self.active[i] and self.done[i]):
                    continue

                if self.success[i] and (self.saved_success < target):
                    ep_idx = int(self.saved_success)
                    prompt = str(cfg.instruction)
                    extra_ep: Dict[str, Any] = {}
                    self.io.finalize_episode(env_id=i, ep_idx=ep_idx, prompt=prompt, extra_ep=extra_ep)
                    self.saved_success += 1
                    print(f"[Saved] episode {ep_idx}. Total saved: {self.saved_success}")
                else:
                    # Common IO does cleanup via finalize_episode; unsuccessful episodes are not written.
                    self.io.finalize_episode(env_id=i, ep_idx=int(self.saved_success), prompt=str(cfg.instruction), success=False)

                self.active[i] = False

                if self.saved_success < target:
                    new_seed = int(cfg.seed + self.attempted)
                    self.start_episodes(np.array([i], dtype=np.int64), seed=new_seed)

            now = time.perf_counter()
            if (now - self._last_log_t) >= 2.0:
                print(
                    f"[collect] active={int(np.sum(self.active))}/{self.B} "
                    f"saved_success={int(self.saved_success)}/{target} attempted={int(self.attempted)}"
                )
                self._last_log_t = now

        print(f"[DONE] saved_success={self.saved_success}/{target}")
        print(f"Saved to: {cfg.save_dir}")

    def close(self) -> None:
        if hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:
                pass
        self.io.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--data_size", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--max_ctrl_steps", type=int, default=None)
    p.add_argument("--action_mode", type=str, default=None)
    args = p.parse_args()

    env_cfg = ArrangeFlowersEnvCfg()
    if args.action_mode is not None:
        env_cfg.action_mode = str(args.action_mode)

    cfg = CollectorCfg(
        save_dir=args.save_dir if args.save_dir is not None else CollectorCfg.save_dir,
        data_size=args.data_size if args.data_size is not None else CollectorCfg.data_size,
        num_envs=args.num_envs if args.num_envs is not None else CollectorCfg.num_envs,
        seed=args.seed if args.seed is not None else CollectorCfg.seed,
        save_video=(not args.no_video),
        max_ctrl_steps=args.max_ctrl_steps if args.max_ctrl_steps is not None else CollectorCfg.max_ctrl_steps,
    )

    runner = ArrangeFlowersCollector(cfg, env_cfg=env_cfg)
    try:
        runner.collect()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
