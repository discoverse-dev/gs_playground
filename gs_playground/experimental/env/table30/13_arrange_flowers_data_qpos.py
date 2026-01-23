# =============================================================================
# File: collect_arrange_flowers.py
# =============================================================================
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
from scipy.spatial.transform import Rotation

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
        if bgr is None:
            return
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
    max_ctrl_steps: int = 1200

    # motion params
    max_dp: float = 0.005
    pos_tol: float = 0.002

    # rotation control
    rot_gain: float = 1.0
    max_dr: float = 0.01
    angle_tol: float = 0.05

    # task offsets
    grasp_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    lift_height_z: float = 0.45

    # vase insertion
    vase_rim_height: float = 0.55
    insert_depth: float = 0.15

    # gripper
    gripper_open: float = 0.0
    gripper_close: float = 0.82

    # timing
    close_hold_steps: int = 20
    waypoint_dwell_steps: int = 15

    # sampling / render
    sample_every_steps: int = 1
    save_video: bool = True
    render_every_steps: int = 1
    video_fps: int = 30
    video_w: int = 640
    video_h: int = 480
    cam_view_key: Optional[str] = "pixels/view_0"

    # IMPORTANT: updated instruction (reflect new yaw rotate stages)
    instruction: str = (
        "Pick up the flower. After aligning XY above the flower and before descending, rotate the wrist "
        "to match the randomized yaw angle. Grasp the flower, lift up, rotate the wrist back to the "
        "original orientation before moving above the vase, then align, insert, release, and retreat."
    )


# -----------------------------------------------------------------------------
# Collector
# -----------------------------------------------------------------------------
class ArrangeFlowersCollector:
    # -------------------------
    # FSM States (UPDATED)
    # -------------------------
    # Phase 1: Approach & Pre-rotate & Grasp
    ST_APP_LIFT_Z = 0
    ST_APP_ALIGN_X = 1
    ST_APP_ALIGN_Y = 2
    ST_APP_ROT_TO_YAW = 3      # NEW: rotate to randomized yaw after XY align, before descend
    ST_APP_DESCEND = 4
    ST_CLOSE = 5

    # Phase 2: Lift & Rotate-back & Transport
    ST_LIFT_HIGH = 6
    ST_ROT_BACK_TO_ORIG = 7    # NEW: rotate back after lift, before moving to vase
    ST_TRP_ALIGN_X = 8
    ST_TRP_ALIGN_Y = 9

    # Phase 3: Fine Rotation (existing two steps near vase)
    ST_ROT_X = 10
    ST_ROT_Z = 11

    # Phase 4: Align & Insert
    ST_ALIGN_POS = 12
    ST_INSERT = 13

    # Phase 5: Release & Home
    ST_OPEN = 14
    ST_RETREAT_Z = 15
    ST_TO_HOME_X = 16
    ST_TO_HOME_Y = 17
    ST_TO_HOME_Z = 18
    ST_DONE = 19

    def __init__(self, cfg: CollectorCfg, env_cfg: Optional[ArrangeFlowersEnvCfg] = None):
        self.cfg = cfg
        os.makedirs(cfg.save_dir, exist_ok=True)
        self.videos_dir = os.path.join(cfg.save_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

        # Env
        self.env_cfg = env_cfg if env_cfg is not None else ArrangeFlowersEnvCfg()
        self.env = ArrangeFlowersEnv(self.env_cfg, num_envs=int(cfg.num_envs))
        self.env.reset()

        self.model = self.env.model
        self.B = int(cfg.num_envs)

        self.flower_body = self.env.flower_body
        self.vase_body = self.env.vase_body
        self.cam_view_key = cfg.cam_view_key or "pixels/view_0"

        # Lifecycle
        self.active = np.zeros(self.B, dtype=bool)
        self.done = np.zeros(self.B, dtype=bool)
        self.success = np.zeros(self.B, dtype=bool)
        self.ctrl_step = np.zeros(self.B, dtype=np.int32)

        # FSM
        self.states = np.zeros(self.B, dtype=np.int32)
        self.state_enter_step = np.zeros(self.B, dtype=np.int32)
        self.state_reach_step = np.full(self.B, -1, dtype=np.int32)

        self._attempt_id = np.zeros(self.B, dtype=np.int64)

        # Control targets (executor)
        self.exec_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.exec_quat = np.zeros((self.B, 4), dtype=np.float32)

        # Latches
        self.latched_flower_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.latched_vase_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.latched_start_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.latched_start_quat = np.zeros((self.B, 4), dtype=np.float32)

        # NEW: randomized yaw (from env info)
        self.latched_yaw_rad = np.zeros((self.B,), dtype=np.float32)

        # NEW: rotation targets for yaw stages
        self.quat_to_yaw = np.zeros((self.B, 4), dtype=np.float32)

        # Existing rotation targets
        self.quat_rot_y = np.zeros((self.B, 4), dtype=np.float32)
        self.quat_final = np.zeros((self.B, 4), dtype=np.float32)

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
            "times": [],
            "logic_states": [],
            "qpos": [],
            "ee_pose": [],
            "gripper": [],
            "ctrl": [],
            "reward": [],
            "is_success": [],
            "video_frames": 0,
        }

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

        # Make sure state/info is populated; if your framework already does this on reset, this is a no-op.
        # If rand_yaw is still missing, we do one zero-action step as fallback.
        if ("rand_yaw_rad" not in self.env._state.info) or (self.env._state.info["rand_yaw_rad"] is None):
            zero_action = np.zeros((self.B, 7), dtype=np.float32)
            self.env.step(zero_action)

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
            if len(pose) == 7:
                self.exec_quat[idx] = pose[3:]
            elif len(pose) == 6:
                euler = pose[3:]
                self.exec_quat[idx] = Rotation.from_euler("xyz", euler).as_quat()
            self.latched_start_quat[idx] = self.exec_quat[idx].copy()

        # Latch object poses
        flower_pose = np.asarray(self.flower_body.get_pose(data), dtype=np.float32)  # (B, 7) or sliced; here full
        vase_pose = np.asarray(self.vase_body.get_pose(data), dtype=np.float32)

        self.latched_flower_pos[env_ids] = flower_pose[env_ids, :3]
        self.latched_vase_pos[env_ids] = vase_pose[env_ids, :3]

        # NEW: read randomized yaw from env info
        info = self.env._state.info
        yaw_rad_all = info.get("rand_yaw_rad", None)
        if yaw_rad_all is None:
            raise RuntimeError("Env state.info missing 'rand_yaw_rad'. Check ArrangeFlowersEnv._compute_reward().")

        self.latched_yaw_rad[env_ids] = -1 *yaw_rad_all[env_ids].astype(np.float32)

        # NEW: build target quat_to_yaw = Rz(yaw) * q_start
        for idx in env_ids:
            q_start = Rotation.from_quat(self.latched_start_quat[idx])
            yaw = float(self.latched_yaw_rad[idx])
            r_yaw = Rotation.from_euler("z", yaw, degrees=False)
            self.quat_to_yaw[idx] = (r_yaw * q_start).as_quat().astype(np.float32)

        # Existing: step-by-step rotation near vase (kept as-is)
        r_z = Rotation.from_euler("z", -35.0, degrees=True)
        for idx in env_ids:
            reachable_euler = np.array([-155, 75, -60])
            q_reachable = Rotation.from_euler("xyz", reachable_euler, degrees=True)
            self.quat_rot_y[idx] = q_reachable.as_quat().astype(np.float32)

            q_final = r_z * q_reachable
            self.quat_final[idx] = q_final.as_quat().astype(np.float32)

        # Start state
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
            try:
                os.remove(tmp_path)
            except Exception:
                pass
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
        if vw is None:
            return
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

        if True:
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
            try:
                os.remove(self._tmp_video_paths[env_id])
            except Exception:
                pass
        self.buffers[env_id] = self._new_buffer()

    def _flush_jsonl(self, env_id: int, ep_idx: int, vid_path: str):
        path = os.path.join(self.cfg.save_dir, f"episode_{ep_idx:05d}.jsonl")
        buf = self.buffers[env_id]
        n = len(buf["times"])
        prompt = self.cfg.instruction
        yaw_deg = float(self.latched_yaw_rad[env_id] * 180.0 / np.pi)

        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                rec = {
                    "images_1": {"url": vid_path, "type": "video", "frame_idx": i},
                    "prompt": prompt,
                    "rand_yaw_deg": yaw_deg,
                    "qpos": buf["qpos"][i],
                    "ee_pose": buf["ee_pose"][i],
                    "gripper": buf["gripper"][i],
                    "ctrl": buf["ctrl"][i],
                    "is_robot": True,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _enter_state(self, mask: np.ndarray, new_state: int) -> None:
        if not np.any(mask):
            return
        self.states[mask] = new_state
        self.state_enter_step[mask] = self.ctrl_step[mask].copy()
        self.state_reach_step[mask] = -1

    def _step_logic(self) -> None:
        cfg = self.cfg
        B = self.B
        obs = self.env._state.obs
        running = self.active & (~self.done)
        if not np.any(running):
            return

        s = self.states

        # --- 1. Define Key Points ---
        start_p = self.latched_start_pos
        flower_p = self.latched_flower_pos
        vase_p = self.latched_vase_pos
        g_off = np.array(cfg.grasp_offset, dtype=np.float32)

        # Approach flower
        p_app_lift = start_p.copy()
        p_app_lift[:, 2] = np.maximum(start_p[:, 2], 0.3)

        p_app_x = p_app_lift.copy()
        p_app_x[:, 0] = flower_p[:, 0] + g_off[0]

        p_app_y = p_app_x.copy()
        p_app_y[:, 1] = flower_p[:, 1] + g_off[1]

        p_grasp = p_app_y.copy()
        p_grasp[:, 2] = flower_p[:, 2] + g_off[2]

        # Lift high
        p_lift_high = p_grasp.copy()
        p_lift_high[:, 2] = cfg.lift_height_z

        # Vase hover
        p_vase_hover = p_lift_high.copy()
        p_vase_hover[:, 0] = vase_p[:, 0]
        p_vase_hover[:, 1] = vase_p[:, 1]

        # Align + insert
        align_off = np.array((0.0, 0.0, 0.1), dtype=np.float32)  # keep your prior default
        p_aligned = p_vase_hover.copy()
        p_aligned += align_off

        p_insert = p_aligned.copy()
        p_insert[:, 2] -= cfg.insert_depth

        # Retreat + home
        p_retreat = p_aligned.copy()
        p_retreat[:, 0] -= 0.2

        home_pos = np.tile(np.array([0.335, 0.0, 0.11], dtype=np.float32), (B, 1))
        p_home_x = p_retreat.copy()
        p_home_x[:, 0] = home_pos[:, 0]
        p_home_y = p_home_x.copy()
        p_home_y[:, 1] = home_pos[:, 1]
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
        q_yaw = self.quat_to_yaw
        q_step1 = self.quat_rot_y
        q_step2 = self.quat_final

        # Phase 1: Approach + rotate-to-yaw + grasp
        set_target(self.ST_APP_LIFT_Z, p_app_lift, q_def, cfg.gripper_open)
        set_target(self.ST_APP_ALIGN_X, p_app_x, q_def, cfg.gripper_open)
        set_target(self.ST_APP_ALIGN_Y, p_app_y, q_def, cfg.gripper_open)

        # NEW: rotate to randomized yaw (hold position)
        set_target(self.ST_APP_ROT_TO_YAW, p_app_y, q_yaw, cfg.gripper_open)

        # descend/grasp with yaw
        set_target(self.ST_APP_DESCEND, p_grasp, q_yaw, cfg.gripper_open)
        set_target(self.ST_CLOSE, p_grasp, q_yaw, cfg.gripper_close)

        # Phase 2: lift + rotate back + move to vase
        set_target(self.ST_LIFT_HIGH, p_lift_high, q_yaw, cfg.gripper_close)
        set_target(self.ST_ROT_BACK_TO_ORIG, p_lift_high, q_def, cfg.gripper_close)
        set_target(self.ST_TRP_ALIGN_X, p_vase_hover, q_def, cfg.gripper_close)
        set_target(self.ST_TRP_ALIGN_Y, p_vase_hover, q_def, cfg.gripper_close)

        # Phase 3/4/5: existing behavior
        set_target(self.ST_ROT_X, p_vase_hover, q_step1, cfg.gripper_close)
        set_target(self.ST_ROT_Z, p_vase_hover, q_step2, cfg.gripper_close)
        set_target(self.ST_ALIGN_POS, p_aligned, q_step2, cfg.gripper_close)
        set_target(self.ST_INSERT, p_insert, q_step2, cfg.gripper_close)
        set_target(self.ST_OPEN, p_insert, q_step2, cfg.gripper_open)
        set_target(self.ST_RETREAT_Z, p_retreat, q_step2, cfg.gripper_open)
        set_target(self.ST_TO_HOME_X, p_home_x, q_def, cfg.gripper_open)
        set_target(self.ST_TO_HOME_Y, p_home_y, q_def, cfg.gripper_open)
        set_target(self.ST_TO_HOME_Z, p_home_z, q_def, cfg.gripper_open)

        # --- Rotation-only states: position compliance (freeze current position) ---
        def freeze_pos_for_state(state_id: int):
            mask = running & (s == state_id)
            if np.any(mask):
                current_pos = obs["ee_pose"][mask, :3]
                tgt_pos_curr[mask] = current_pos
                self.exec_pos[mask] = current_pos

        freeze_pos_for_state(self.ST_APP_ROT_TO_YAW)
        freeze_pos_for_state(self.ST_ROT_BACK_TO_ORIG)
        freeze_pos_for_state(self.ST_ROT_X)
        freeze_pos_for_state(self.ST_ROT_Z)

        # --- 3. Execution ---
        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos_curr, cfg.max_dp)

        # Incremental rotation step
        for i in range(B):
            if not running[i]:
                continue
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

        # Action (PD in eef_relative)
        ref_pose_6d = self.env.robot.ref_ee_pose
        ref_pos = ref_pose_6d[:, :3]
        ref_euler = ref_pose_6d[:, 3:6]
        ref_quat = Rotation.from_euler("xyz", ref_euler, degrees=False).as_quat().astype(np.float32)

        pos_err = (self.exec_pos - ref_pos) * 0.5

        rotvec_cmd = np.zeros((B, 3), dtype=np.float32)
        for i in range(B):
            if not running[i]:
                continue
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

        def is_rot_reached(target_q: np.ndarray, use_real_obs: bool = False) -> np.ndarray:
            errs = np.zeros((B,), dtype=np.float32)
            for i in range(B):
                if use_real_obs:
                    rot_data = obs["ee_pose"][i, 3:]
                    if len(rot_data) == 3:
                        r1 = Rotation.from_euler("xyz", rot_data, degrees=False)
                    elif len(rot_data) == 4:
                        r1 = Rotation.from_quat(rot_data)
                    else:
                        r1 = Rotation.identity()
                else:
                    r1 = Rotation.from_quat(self.exec_quat[i])

                r2 = Rotation.from_quat(target_q[i])
                dq = r1 * r2.inv()
                errs[i] = np.linalg.norm(dq.as_rotvec())
            return errs < cfg.angle_tol

        def _check_and_dwell(
            state_id: int,
            pos_tgt: np.ndarray,
            rot_tgt: np.ndarray,
            use_rot: bool = False,
            check_real_rot: bool = False,
        ) -> np.ndarray:
            in_state = running & (s == state_id)

            p_ok = is_pos_reached(pos_tgt)

            if use_rot:
                r_ok = is_rot_reached(rot_tgt, use_real_obs=check_real_rot)
                reached = r_ok if check_real_rot else (p_ok & r_ok)
            else:
                reached = p_ok

            just_reached = in_state & reached & (self.state_reach_step == -1)
            if np.any(just_reached):
                self.state_reach_step[just_reached] = self.ctrl_step[just_reached]

            has_reached = self.state_reach_step != -1
            dwell_pass = (self.ctrl_step - self.state_reach_step) >= int(cfg.waypoint_dwell_steps)

            return in_state & has_reached & dwell_pass & reached

        # Phase 1: Approach -> XY align -> rotate to yaw -> descend -> close
        self._enter_state(_check_and_dwell(self.ST_APP_LIFT_Z, p_app_lift, q_def), self.ST_APP_ALIGN_X)
        self._enter_state(_check_and_dwell(self.ST_APP_ALIGN_X, p_app_x, q_def), self.ST_APP_ALIGN_Y)

        # NEW: XY aligned -> rotate to yaw
        self._enter_state(_check_and_dwell(self.ST_APP_ALIGN_Y, p_app_y, q_def), self.ST_APP_ROT_TO_YAW)

        # NEW: rotate-to-yaw completed -> descend
        self._enter_state(
            _check_and_dwell(self.ST_APP_ROT_TO_YAW, self.exec_pos, q_yaw, use_rot=True, check_real_rot=True),
            self.ST_APP_DESCEND,
        )

        self._enter_state(_check_and_dwell(self.ST_APP_DESCEND, p_grasp, q_yaw), self.ST_CLOSE)

        # Close hold -> lift
        mask_close = running & (s == self.ST_CLOSE)
        if np.any(mask_close):
            t_in = self.ctrl_step - self.state_enter_step
            done_close = t_in >= cfg.close_hold_steps
            self._enter_state(mask_close & done_close, self.ST_LIFT_HIGH)

        # Phase 2: lift -> rotate back -> move to vase
        self._enter_state(_check_and_dwell(self.ST_LIFT_HIGH, p_lift_high, q_yaw), self.ST_ROT_BACK_TO_ORIG)

        self._enter_state(
            _check_and_dwell(self.ST_ROT_BACK_TO_ORIG, self.exec_pos, q_def, use_rot=True, check_real_rot=True),
            self.ST_TRP_ALIGN_X,
        )

        self._enter_state(_check_and_dwell(self.ST_TRP_ALIGN_X, p_vase_hover, q_def), self.ST_TRP_ALIGN_Y)
        self._enter_state(_check_and_dwell(self.ST_TRP_ALIGN_Y, p_vase_hover, q_def), self.ST_ROT_X)

        # Phase 3: near-vase rotation sequence (existing)
        self._enter_state(
            _check_and_dwell(self.ST_ROT_X, self.exec_pos, q_step1, use_rot=True, check_real_rot=True),
            self.ST_ROT_Z,
        )
        self._enter_state(
            _check_and_dwell(self.ST_ROT_Z, self.exec_pos, q_step2, use_rot=True, check_real_rot=True),
            self.ST_ALIGN_POS,
        )

        # Phase 4: align + insert
        self._enter_state(_check_and_dwell(self.ST_ALIGN_POS, p_aligned, q_step2, use_rot=True), self.ST_INSERT)
        self._enter_state(_check_and_dwell(self.ST_INSERT, p_insert, q_step2, use_rot=True), self.ST_OPEN)

        # Phase 5: open + retreat + home
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
                if not running[i]:
                    continue
                fsm_done = self.states[i] == self.ST_DONE
                timeout = self.ctrl_step[i] >= int(cfg.max_ctrl_steps)
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
            if vw:
                vw.close()


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
