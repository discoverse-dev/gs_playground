
# =============================================================================
# File: 02_stack_color_blocks_data_qpos_refactored.py
#
# Notes:
#   - This script keeps ONLY: task FSM + keypoint computation + control logic.
#   - Video, buffering, JSONL export, and pose fixes are handled by table30_collect_common.py
#   - GS (pixels/...) dual-view videos are saved; Motrix video is NOT saved.
# =============================================================================
from __future__ import annotations

import os
import time
import json
import shutil
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from gs_playground.src.manipulation.tasks.table30._02_stack_color_blocks_franka import (
    StackColorBlocksEnv,
    StackColorBlocksEnvCfg,
)

from table30_collect_common import (
    smooth_step_pos,
    normalize_quat,
    wrap_to_pi,
    quat_to_yaw,
    VideoCfg,
    PoseFixCfg,
    Table30CollectorIO,
)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CollectorCfg:
    # dataset
    data_size: int = 3
    num_envs: int = 3
    seed: int = 42
    save_dir: str = "./data/table30_stack_color_blocks_collect_dual_view"

    # env control
    max_ctrl_steps: int = 800

    # motion params
    max_dp: float = 0.005
    pos_tol: float = 0.001

    # rotation control
    rot_gain: float = 0.6
    max_dr: float = 0.08
    yaw_tol: float = 0.03

    yaw_offset_grasp: float = 0.0
    yaw_offset_place: float = 0.0

    # task specific offsets
    grasp_down_z: float = 0.0
    lift_dz: float = 0.06
    cube_half: float = 0.025

    # gripper
    gripper_open: float = 0.0
    gripper_close: float = 0.82

    # timing / dwell
    close_hold_steps: int = 15
    stack_hold_steps: int = 10
    waypoint_dwell_steps: int = 20

    # sampling / render
    sample_every_steps: int = 1
    save_video: bool = True
    render_every_steps: int = 1
    video_fps: int = 30
    video_w: int = 640
    video_h: int = 480

    # Dual-view GS videos
    cam_view_key: Sequence[str] = field(default_factory=lambda: ["pixels/view_0", "pixels/view_1"])

    # JSONL pose export fix (same as your previous scripts)
    export_z_offset: float = 0.1525
    export_yaw_offset: float = -0.5 * np.pi
    export_yaw_sign: float = -1.0

    # text fields
    subtask: Optional[str] = "Stack specific colored blocks."


# -----------------------------------------------------------------------------
# Collector (FSM + keypoints + control only)
# -----------------------------------------------------------------------------
class StackColorBlocksCollector:
    # Phase 1: Approach (Manhattan)
    ST_APP_LIFT_Z = 0
    ST_APP_ALIGN_X = 1
    ST_APP_ALIGN_Y = 2
    ST_APP_ALIGN_YAW = 3
    ST_APP_DESCEND = 4

    # Phase 2: Grasp
    ST_CLOSE = 5

    # Phase 3: Transport (Manhattan)
    ST_TRP_LIFT_Z = 6
    ST_TRP_ALIGN_X = 7
    ST_TRP_ALIGN_Y = 8
    ST_TRP_ALIGN_YAW = 9
    ST_TO_STACK = 10

    # Phase 4: Release & End
    ST_OPEN_HOLD = 11
    ST_RETREAT = 12
    ST_TO_HOME = 13
    ST_DONE = 14

    def __init__(self, cfg: CollectorCfg, env_cfg: Optional[StackColorBlocksEnvCfg] = None):
        self.cfg = cfg
        os.makedirs(cfg.save_dir, exist_ok=True)

        # Env
        self.env_cfg = env_cfg if env_cfg is not None else StackColorBlocksEnvCfg()
        self.env = StackColorBlocksEnv(self.env_cfg, num_envs=int(cfg.num_envs))
        self.env.reset()

        self.B = int(cfg.num_envs)

        # Bodies
        self.cube_names = self.env_cfg.cube_names
        self.cube_bodies = self.env.cube_bodies

        # Cameras (GS only)
        self.cam_keys = list(cfg.cam_view_key)

        # IO (video/buffer/jsonl/pose fix)
        pose_fix = PoseFixCfg(
            z_offset=float(cfg.export_z_offset),
            yaw_offset=float(cfg.export_yaw_offset),
            yaw_sign=float(cfg.export_yaw_sign),
            wrap_yaw=True,
        )
        video_cfg = VideoCfg(fps=int(cfg.video_fps), width=int(cfg.video_w), height=int(cfg.video_h), codec="h264")
        self.io = Table30CollectorIO(
            save_dir=str(cfg.save_dir),
            num_envs=self.B,
            cam_keys=self.cam_keys,
            save_video=bool(cfg.save_video),
            sample_every_steps=int(cfg.sample_every_steps),
            render_every_steps=int(cfg.render_every_steps),
            video_cfg=video_cfg,
            pose_fix=pose_fix,
            dt=0.02,
        )

        # lifecycle
        self.active = np.zeros(self.B, dtype=bool)
        self.done = np.zeros(self.B, dtype=bool)
        self.success = np.zeros(self.B, dtype=bool)
        self.ctrl_step = np.zeros(self.B, dtype=np.int32)

        # FSM
        self.states = np.zeros(self.B, dtype=np.int32)
        self.state_enter_step = np.zeros(self.B, dtype=np.int32)
        self.state_reach_step = np.full(self.B, -1, dtype=np.int32)

        self._attempt_id = np.zeros(self.B, dtype=np.int64)

        # task-specific
        self.top_idx = np.zeros(self.B, dtype=np.int32)
        self.base_idx = np.zeros(self.B, dtype=np.int32)
        self.stack_hold_counter = np.zeros(self.B, dtype=np.int32)

        # control targets
        self.exec_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.exec_quat = np.zeros((self.B, 4), dtype=np.float32)  # xyzw

        # latches
        self.latched_start_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.latched_start_quat = np.zeros((self.B, 4), dtype=np.float32)
        self.latched_start_yaw = np.zeros((self.B,), dtype=np.float32)

        self.latched_top_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.latched_base_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.latched_top_yaw = np.zeros((self.B,), dtype=np.float32)
        self.latched_base_yaw = np.zeros((self.B,), dtype=np.float32)

        # per env prompt (changes each episode based on sampled cubes)
        self.ep_prompt = np.array([""] * self.B, dtype=object)

        self.saved_count = 0
        self.attempted = 0
        self._last_log_t = time.perf_counter()
        self._last_action = np.zeros((self.B, 7), dtype=np.float32)

    # -----------------------------
    # Episode init / reset
    # -----------------------------
    def start_episodes(self, env_ids: np.ndarray, seed: int) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if env_ids.size == 0:
            return

        # seed env rng best-effort
        try:
            self.env._rng = np.random.default_rng(int(seed))
        except Exception:
            pass

        done_mask = np.zeros((self.B,), dtype=bool)
        done_mask[env_ids] = True
        self.env.reset(done=done_mask)

        # lifecycle reset
        self.active[env_ids] = True
        self.done[env_ids] = False
        self.success[env_ids] = False
        self.ctrl_step[env_ids] = 0
        self.state_enter_step[env_ids] = 0
        self.state_reach_step[env_ids] = -1
        self.stack_hold_counter[env_ids] = 0

        data = self.env._state.data
        self.env.robot.reset_envs(data, done_mask)
        self.env.robot.update_reference(data)

        # latch start EE pose
        all_poses = self.env.robot.get_ee_pose(data)
        for idx in env_ids.tolist():
            pose = all_poses[idx]
            self.exec_pos[idx] = pose[:3]
            self.latched_start_pos[idx] = pose[:3]

            if len(pose) == 7:
                q = pose[3:]
            elif len(pose) == 6:
                q = Rotation.from_euler("xyz", pose[3:6], degrees=False).as_quat().astype(np.float32)
            else:
                q = Rotation.identity().as_quat().astype(np.float32)

            q = normalize_quat(np.asarray(q, dtype=np.float32)[None, :])[0]
            self.exec_quat[idx] = q
            self.latched_start_quat[idx] = q.copy()
            self.latched_start_yaw[idx] = Rotation.from_quat(q).as_euler("xyz", degrees=False)[2].astype(np.float32)

        # latch sampled cubes (from env)
        self.top_idx[env_ids] = np.asarray(self.env.top_idx[env_ids], dtype=np.int32)
        self.base_idx[env_ids] = np.asarray(self.env.base_idx[env_ids], dtype=np.int32)

        # latch cube pose + yaw
        cube_pose = np.stack([np.asarray(b.get_pose(data), dtype=np.float32) for b in self.cube_bodies], axis=1)
        self.latched_top_pos[env_ids] = cube_pose[env_ids, self.top_idx[env_ids], :3]
        self.latched_base_pos[env_ids] = cube_pose[env_ids, self.base_idx[env_ids], :3]

        top_q = cube_pose[env_ids, self.top_idx[env_ids], 3:7]
        base_q = cube_pose[env_ids, self.base_idx[env_ids], 3:7]
        top_yaw = Rotation.from_quat(top_q).as_euler("xyz", degrees=False)[:, 2]
        base_yaw = Rotation.from_quat(base_q).as_euler("xyz", degrees=False)[:, 2]
        # keep your previous sign convention
        self.latched_top_yaw[env_ids] = top_yaw.astype(np.float32) * -1.0
        self.latched_base_yaw[env_ids] = base_yaw.astype(np.float32) * -1.0

        # prompt per env episode
        for idx in env_ids.tolist():
            t_raw = str(self.cube_names[int(self.top_idx[idx])])
            b_raw = str(self.cube_names[int(self.base_idx[idx])])
            t_name = t_raw.replace("cube_", "").lower()
            b_name = b_raw.replace("cube_", "").lower()
            self.ep_prompt[idx] = f"Stack the {t_name} block on top of the {b_name} block."

        # start state
        self.states[env_ids] = self.ST_APP_LIFT_Z

        # IO reset per env
        for env_id in env_ids.tolist():
            self.io.reset_env(env_id)
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
    # Core logic (FSM + keypoints + control)
    # -----------------------------
    def _check_stack_success(self, mask: np.ndarray) -> np.ndarray:
        data = self.env._state.data
        cube_pose = np.stack([np.asarray(b.get_pose(data), dtype=np.float32) for b in self.cube_bodies], axis=1)
        row_ids = np.arange(self.B)

        t_idx = self.top_idx
        b_idx = self.base_idx
        tp = cube_pose[row_ids, t_idx, :3]
        bp = cube_pose[row_ids, b_idx, :3]

        xy_dist = np.linalg.norm(tp[:, :2] - bp[:, :2], axis=1)
        z_diff = tp[:, 2] - bp[:, 2]
        target_z = 2.0 * float(self.cfg.cube_half)

        xy_ok = xy_dist < 0.03
        z_ok = np.abs(z_diff - target_z) < 0.02
        return (xy_ok & z_ok) & mask

    def _step_logic(self) -> None:
        cfg = self.cfg
        B = self.B

        running = self.active & (~self.done)
        if not np.any(running):
            return

        # 1) keypoints
        start_p = self.latched_start_pos
        top_p = self.latched_top_pos
        base_p = self.latched_base_pos

        safe_z = top_p[:, 2] + float(cfg.lift_dz)

        p_app_lift_z = start_p.copy()
        p_app_lift_z[:, 2] = safe_z

        p_app_align_x = p_app_lift_z.copy()
        p_app_align_x[:, 0] = top_p[:, 0]

        p_app_align_y = top_p.copy()
        p_app_align_y[:, 2] = safe_z

        p_grasp = top_p + np.array([0.0, 0.0, float(cfg.grasp_down_z)], dtype=np.float32)

        p_trp_lift_z = top_p.copy()
        p_trp_lift_z[:, 2] = safe_z

        p_trp_align_x = p_trp_lift_z.copy()
        p_trp_align_x[:, 0] = base_p[:, 0] - 0.015

        p_trp_align_y = base_p.copy()
        p_trp_align_y[:, 2] = safe_z
        p_trp_align_y[:, 0] = base_p[:, 0] - 0.015

        p_stack = base_p + np.array([0.0, 0.0, 2.0 * float(cfg.cube_half) + 0.005], dtype=np.float32)
        p_stack[:, 0] = base_p[:, 0] - 0.015

        p_retreat = base_p + np.array([0.0, 0.0, float(cfg.lift_dz) + 0.10], dtype=np.float32)

        p_home = np.tile(np.array([0.33502, 0.0, 0.11], dtype=np.float32), (B, 1))

        # 2) targets by state
        s = self.states
        tgt_pos = self.exec_pos.copy()
        grip_cmd = np.full((B,), float(cfg.gripper_open), dtype=np.float32)

        tgt_yaw = np.full((B,), np.nan, dtype=np.float32)

        def set_target(state_id: int, pos: np.ndarray, grip: float):
            m = running & (s == state_id)
            if np.any(m):
                tgt_pos[m] = pos[m]
                grip_cmd[m] = float(grip)

        def set_target_yaw(state_id: int, pos: np.ndarray, grip: float, yaw_arr: np.ndarray):
            m = running & (s == state_id)
            if np.any(m):
                tgt_pos[m] = pos[m]
                grip_cmd[m] = float(grip)
                tgt_yaw[m] = yaw_arr[m]

        set_target(self.ST_APP_LIFT_Z, p_app_lift_z, cfg.gripper_open)
        set_target(self.ST_APP_ALIGN_X, p_app_align_x, cfg.gripper_open)
        set_target(self.ST_APP_ALIGN_Y, p_app_align_y, cfg.gripper_open)
        set_target_yaw(self.ST_APP_ALIGN_YAW, p_app_align_y, cfg.gripper_open, self.latched_top_yaw + float(cfg.yaw_offset_grasp))
        set_target(self.ST_APP_DESCEND, p_grasp, cfg.gripper_open)

        set_target(self.ST_CLOSE, p_grasp, cfg.gripper_close)

        set_target(self.ST_TRP_LIFT_Z, p_trp_lift_z, cfg.gripper_close)
        set_target(self.ST_TRP_ALIGN_X, p_trp_align_x, cfg.gripper_close)
        set_target(self.ST_TRP_ALIGN_Y, p_trp_align_y, cfg.gripper_close)
        set_target_yaw(self.ST_TRP_ALIGN_YAW, p_trp_align_y, cfg.gripper_close, self.latched_base_yaw + float(cfg.yaw_offset_place))
        set_target(self.ST_TO_STACK, p_stack, cfg.gripper_close)

        set_target(self.ST_OPEN_HOLD, p_stack, cfg.gripper_open)
        set_target(self.ST_RETREAT, p_retreat, cfg.gripper_open)
        set_target_yaw(self.ST_TO_HOME, p_home, cfg.gripper_open, self.latched_start_yaw)

        # 3) control
        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos, float(cfg.max_dp))

        # yaw control only in yaw states
        want_rot = running & (~np.isnan(tgt_yaw)) & (
            (s == self.ST_APP_ALIGN_YAW) | (s == self.ST_TRP_ALIGN_YAW) | (s == self.ST_TO_HOME)
        )

        # reference pose
        ref_pose = self.env.robot.ref_ee_pose
        ref_pos = ref_pose[:, :3]
        ref_euler = ref_pose[:, 3:6]
        ref_quat = Rotation.from_euler("xyz", ref_euler, degrees=False).as_quat().astype(np.float32)

        # current ee yaw from obs
        obs = self.env._state.obs
        ee_pose = obs.get("ee_pose", None)
        if ee_pose is not None and ee_pose.shape[1] == 7:
            ee_quat = ee_pose[:, 3:7].astype(np.float32)
        elif ee_pose is not None and ee_pose.shape[1] == 6:
            ee_quat = Rotation.from_euler("xyz", ee_pose[:, 3:6], degrees=False).as_quat().astype(np.float32)
        else:
            ee_quat = ref_quat.copy()
        curr_yaw = Rotation.from_quat(ee_quat).as_euler("xyz", degrees=False)[:, 2].astype(np.float32)

        desired_quat = self.exec_quat.copy()
        if np.any(want_rot):
            # make target yaw closest to current yaw
            dy = wrap_to_pi(tgt_yaw[want_rot] - curr_yaw[want_rot])
            target_yaw = curr_yaw[want_rot] + dy

            # 90-degree symmetry for cubes
            symmetry = np.pi / 2.0
            delta = target_yaw - self.latched_start_yaw[want_rot]
            delta = (delta + symmetry / 2.0) % symmetry - symmetry / 2.0

            r_delta = Rotation.from_euler("z", delta, degrees=False)
            r_start = Rotation.from_quat(self.latched_start_quat[want_rot])
            r_target = r_delta * r_start
            desired_quat[want_rot] = r_target.as_quat().astype(np.float32)
            self.exec_quat[want_rot] = desired_quat[want_rot]

        rotvec_cmd = np.zeros((B, 3), dtype=np.float32)
        if np.any(want_rot):
            r_des = Rotation.from_quat(desired_quat[want_rot])
            r_ref = Rotation.from_quat(ref_quat[want_rot])
            r_err = r_des * r_ref.inv()
            rv = r_err.as_rotvec().astype(np.float32)
            mag = np.linalg.norm(rv, axis=1, keepdims=True) + 1e-9
            scale = np.minimum(1.0, float(cfg.max_dr) / mag)
            rv = rv * scale
            rotvec_cmd[want_rot] = rv * float(cfg.rot_gain)

        action = np.zeros((B, 7), dtype=np.float32)
        action[:, :3] = (self.exec_pos - ref_pos) * 0.5
        action[:, 3:6] = rotvec_cmd
        action[:, 6] = grip_cmd
        self._last_action[:] = action

        self.env.step(action)

        # 4) transitions
        def is_reached(p: np.ndarray) -> np.ndarray:
            return np.linalg.norm(self.exec_pos - p, axis=1) < float(cfg.pos_tol)

        def _check_reach_dwell(state_id: int, p: np.ndarray) -> np.ndarray:
            in_s = running & (self.states == state_id)
            ok = is_reached(p)
            just = in_s & ok & (self.state_reach_step == -1)
            if np.any(just):
                self.state_reach_step[just] = self.ctrl_step[just]
            dwell = (self.ctrl_step - self.state_reach_step) >= int(cfg.waypoint_dwell_steps)
            return in_s & (self.state_reach_step != -1) & dwell & ok

        def _check_yaw_dwell(state_id: int, target_yaw: np.ndarray) -> np.ndarray:
            in_s = running & (self.states == state_id)
            dy = wrap_to_pi(curr_yaw - target_yaw)
            symmetry = np.pi / 2.0
            dy = (dy + symmetry / 2.0) % symmetry - symmetry / 2.0
            ok = np.abs(dy) < float(cfg.yaw_tol)
            just = in_s & ok & (self.state_reach_step == -1)
            if np.any(just):
                self.state_reach_step[just] = self.ctrl_step[just]
            dwell = (self.ctrl_step - self.state_reach_step) >= int(cfg.waypoint_dwell_steps)
            return in_s & (self.state_reach_step != -1) & dwell & ok

        self._enter_state(_check_reach_dwell(self.ST_APP_LIFT_Z, p_app_lift_z), self.ST_APP_ALIGN_X)
        self._enter_state(_check_reach_dwell(self.ST_APP_ALIGN_X, p_app_align_x), self.ST_APP_ALIGN_Y)
        self._enter_state(_check_reach_dwell(self.ST_APP_ALIGN_Y, p_app_align_y), self.ST_APP_ALIGN_YAW)

        grasp_yaw = (self.latched_top_yaw + float(cfg.yaw_offset_grasp)).astype(np.float32)
        self._enter_state(_check_yaw_dwell(self.ST_APP_ALIGN_YAW, grasp_yaw), self.ST_APP_DESCEND)

        self._enter_state(_check_reach_dwell(self.ST_APP_DESCEND, p_grasp), self.ST_CLOSE)

        mask_close = running & (self.states == self.ST_CLOSE)
        if np.any(mask_close):
            done_close = (self.ctrl_step - self.state_enter_step) >= int(cfg.close_hold_steps)
            self._enter_state(mask_close & done_close, self.ST_TRP_LIFT_Z)

        self._enter_state(_check_reach_dwell(self.ST_TRP_LIFT_Z, p_trp_lift_z), self.ST_TRP_ALIGN_X)
        self._enter_state(_check_reach_dwell(self.ST_TRP_ALIGN_X, p_trp_align_x), self.ST_TRP_ALIGN_Y)
        self._enter_state(_check_reach_dwell(self.ST_TRP_ALIGN_Y, p_trp_align_y), self.ST_TRP_ALIGN_YAW)

        place_yaw = (self.latched_base_yaw + float(cfg.yaw_offset_place)).astype(np.float32)
        self._enter_state(_check_yaw_dwell(self.ST_TRP_ALIGN_YAW, place_yaw), self.ST_TO_STACK)

        self._enter_state(_check_reach_dwell(self.ST_TO_STACK, p_stack), self.ST_OPEN_HOLD)

        mask_open = running & (self.states == self.ST_OPEN_HOLD)
        if np.any(mask_open):
            is_stacked = self._check_stack_success(mask_open)
            self.stack_hold_counter[mask_open & is_stacked] += 1
            self.stack_hold_counter[mask_open & (~is_stacked)] = 0
            ready = self.stack_hold_counter >= int(cfg.stack_hold_steps)
            self._enter_state(mask_open & ready, self.ST_RETREAT)

        self._enter_state(running & (self.states == self.ST_RETREAT) & is_reached(p_retreat), self.ST_TO_HOME)
        self._enter_state(running & (self.states == self.ST_TO_HOME) & is_reached(p_home), self.ST_DONE)

    # -----------------------------
    # Collection loop (minimal; IO in common)
    # -----------------------------
    def collect(self) -> None:
        cfg = self.cfg
        target_n = int(cfg.data_size)

        all_ids = np.arange(self.B, dtype=np.int64)
        self.start_episodes(all_ids, seed=int(cfg.seed))

        print(f"Starting StackColorBlocks collection. Target: {target_n}")

        while self.saved_count < target_n:
            self._step_logic()

            running = self.active & (~self.done)

            # capture (before video render) to keep frame alignment
            sample_mask = running & ((self.ctrl_step % int(cfg.sample_every_steps)) == 0)
            obs = self.env._state.obs
            for env_id in np.where(sample_mask)[0].tolist():
                extra = {"top_idx": int(self.top_idx[env_id]), "base_idx": int(self.base_idx[env_id])}
                self.io.capture_step(
                    env_id=int(env_id),
                    ctrl_step=int(self.ctrl_step[env_id]),
                    state=int(self.states[env_id]),
                    obs=obs,
                    last_action=self._last_action,
                    success=bool(self.success[env_id]),
                    reward=float(1.0 if self.success[env_id] else 0.0),
                    extra_step=extra,
                )

            # render videos
            if cfg.save_video:
                render_ids = np.where(running)[0].tolist()
                self.io.maybe_write_video(obs, render_ids, int(self.ctrl_step[0]))

            self.ctrl_step[running] += 1

            # termination
            for i in range(self.B):
                if not running[i]:
                    continue
                fsm_done = (int(self.states[i]) == int(self.ST_DONE))
                timeout = (int(self.ctrl_step[i]) >= int(cfg.max_ctrl_steps))
                if fsm_done or timeout:
                    self.done[i] = True
                    is_stacked = bool(self._check_stack_success(np.eye(self.B, dtype=bool)[i])[i])
                    self.success[i] = bool(fsm_done and is_stacked)

            # finalize & restart
            for i in range(self.B):
                if self.active[i] and self.done[i]:
                    if self.success[i] and self.saved_count < target_n:
                        ep_idx = int(self.saved_count)
                        self.io.finalize_episode(
                            env_id=i,
                            ep_idx=ep_idx,
                            prompt=str(self.ep_prompt[i]),
                            success=True,
                            extra_episode={"subtask": str(cfg.subtask or "")},
                        )
                        self.saved_count += 1
                        print(f"[Saved] Episode {ep_idx}. Total saved: {self.saved_count}")
                    else:
                        # cleanup tmp video/buffer
                        self.io.finalize_episode(env_id=i, ep_idx=int(self.saved_count), prompt=str(self.ep_prompt[i]), success=False)

                    if self.saved_count < target_n:
                        self.active[i] = False
                        self.done[i] = False
                        new_seed = int(cfg.seed + self.attempted)
                        self.start_episodes(np.array([i]), seed=new_seed)
                    else:
                        self.active[i] = False

            now = time.perf_counter()
            if (now - self._last_log_t) > 2.0:
                print(f"[Collect] Saved: {self.saved_count}/{target_n} | Active: {int(self.active.sum())}")
                self._last_log_t = now

        print(f"Done. Saved to {cfg.save_dir}")
        self.close()

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
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--data_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no_video", action="store_true")

    args = p.parse_args()

    cfg = CollectorCfg(
        save_dir=args.save_dir if args.save_dir else CollectorCfg.save_dir,
        num_envs=args.num_envs if args.num_envs is not None else CollectorCfg.num_envs,
        data_size=args.data_size if args.data_size is not None else CollectorCfg.data_size,
        seed=args.seed if args.seed is not None else CollectorCfg.seed,
        save_video=(not args.no_video),
    )

    runner = StackColorBlocksCollector(cfg)
    try:
        runner.collect()
    except KeyboardInterrupt:
        pass
    finally:
        runner.close()


if __name__ == "__main__":
    main()
