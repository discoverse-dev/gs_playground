from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from gs_playground.src.manipulation.tasks.table30._13_arrange_flowers import (
    ArrangeFlowersEnv,
    ArrangeFlowersEnvCfg,
)

from table30_collect_common import (
    PoseFixCfg,
    VideoCfg,
    Table30CollectorIO,
    smooth_step_pos,
    normalize_quat,
    wrap_to_pi,
    quat_to_yaw,
)


@dataclass(frozen=True)
class CollectorCfg:
    # dataset
    data_size: int = 5
    num_envs: int = 5
    seed: int = 0
    save_dir: str = "./data/table30_arrange_flowers_refactored"

    # env control
    max_ctrl_steps: int = 2000

    # motion params
    max_dp: float = 0.005
    pos_tol: float = 0.002

    # rotation control (yaw-only task but we still command rotvec)
    rot_gain: float = 1.0
    max_dr: float = 0.01
    yaw_tol: float = 0.08              # rad, for yaw-only reach check
    rot_dwell_steps: int = 15
    rot_state_timeout_steps: int = 250

    # task offsets
    grasp_offset: Tuple[float, float, float] = (-0.03, -0.02, 0.0)
    lift_height_z: float = 0.45

    # insertion
    insert_depth: float = 0.25
    align_z_offset: float = 0.10

    # pregrasp margin (avoid pushing flower before grasp)
    pregrasp_z_margin: float = 0.03

    # gripper
    gripper_open: float = 0.0
    gripper_close: float = 0.82

    # timing
    close_hold_steps: int = 20
    waypoint_dwell_steps: int = 15

    # retreat
    retreat_dx: float = 0.08

    # yaw interfaces (degrees)
    pick1_yaw_deg: float = -30.0
    pick2_yaw_deg: float = -30.0

    # -----------------------------
    # Beautified offsets
    # -----------------------------
    pick1_coarse_xy_offset: Tuple[float, float] = (-0.03, 0.0)
    place1_dst_hover_xy_offset: Tuple[float, float] = (0.03, 0.02)
    pick2_grasp_z_extra: float = 0.05
    place2_src_hover_xy_offset: Tuple[float, float] = (0.05, 0.03)
    place2_insert_z_extra: float = -0.08

    # action scaling
    action_pos_scale: float = 0.5
    action_grip_scale: float = 1.0

    # -----------------------------
    # Video / multi-view saving (GS only)
    # -----------------------------
    sample_every_steps: int = 1
    save_video: bool = True
    render_every_steps: int = 1
    video_fps: int = 30
    video_w: int = 640
    video_h: int = 480
    cam_view_key: Sequence[str] = field(default_factory=lambda: ["pixels/view_0", "pixels/view_1"])

    # MotrixSim renderer video (debug only; default off)
    enable_motrix_video: bool = False
    motrix_video_fps: int = 30
    motrix_video_width: int = 640
    motrix_video_height: int = 480

    # prompt/instruction
    instruction: str = (
        "Pick up the flower from the source vase, place it into the target vase, "
        "then pick it up again and place it back into the source vase. "
        "Each motion first aligns XY, then performs Z motion. Rotation only changes yaw. "
        "Before grasp, do a pregrasp descend then fine-align (Z->Y->X), then final descend and grasp."
    )


class ArrangeFlowersCollector:
    # Pick from source vase (pick1)
    ST_APP1_LIFT_Z = 0
    ST_APP1_ALIGN_X = 1
    ST_APP1_ALIGN_Y = 2
    ST_ROT_PICK1 = 3
    ST_PRE1_Z = 4
    ST_PRE1_ALIGN_Y = 5
    ST_PRE1_ALIGN_X = 6
    ST_DESCEND1 = 7
    ST_CLOSE1 = 8
    ST_LIFT1 = 9

    # Place into target vase (place1)
    ST_TRP1_ALIGN_X = 10
    ST_TRP1_ALIGN_Y = 11
    ST_PLACE1_DESCEND = 12
    ST_OPEN1 = 13
    ST_RETREAT1 = 14

    # Pick again (pick2) from target vase
    ST_APP2_LIFT_Z = 15
    ST_APP2_ALIGN_X = 16
    ST_APP2_ALIGN_Y = 17
    ST_ROT_PICK2 = 18
    ST_PRE2_Z = 19
    ST_PRE2_ALIGN_Y = 20
    ST_PRE2_ALIGN_X = 21
    ST_DESCEND2 = 22
    ST_CLOSE2 = 23
    ST_LIFT2 = 24

    # Place back into source vase (place2)
    ST_TRP2_ALIGN_X = 25
    ST_TRP2_ALIGN_Y = 26
    ST_PLACE2_DESCEND = 27
    ST_OPEN2 = 28
    ST_RETREAT2 = 29

    ST_DONE = 30

    def __init__(self, cfg: CollectorCfg, env_cfg: Optional[ArrangeFlowersEnvCfg] = None):
        self.cfg = cfg
        os.makedirs(cfg.save_dir, exist_ok=True)

        self.env_cfg = env_cfg if env_cfg is not None else ArrangeFlowersEnvCfg()

        # Best-effort: pass MotrixSim recorder config into env cfg (debug only)
        setattr(self.env_cfg, "enable_motrix_video", bool(cfg.enable_motrix_video))
        setattr(self.env_cfg, "motrix_video_fps", int(cfg.motrix_video_fps))
        setattr(self.env_cfg, "motrix_video_width", int(cfg.motrix_video_width))
        setattr(self.env_cfg, "motrix_video_height", int(cfg.motrix_video_height))
        setattr(self.env_cfg, "motrix_video_output_dir", os.path.join(cfg.save_dir, "videos"))

        self.env = ArrangeFlowersEnv(self.env_cfg, num_envs=int(cfg.num_envs))
        self.env.reset()

        self.B = int(cfg.num_envs)

        # bodies
        self.flower_body = self.env.flower_body
        self.vase_src_body = getattr(self.env, "vase_src_body", getattr(self.env, "vase_body", None))
        if self.vase_src_body is None:
            raise RuntimeError("Env missing vase_src_body/vase_body")
        self.vase_dst_body = getattr(self.env, "vase_dst_body", None)
        if self.vase_dst_body is None:
            raise RuntimeError("Env missing vase_dst_body (vase2). Please update env to include vase2.")

        # cameras
        self.cam_keys = list(cfg.cam_view_key)

        # lifecycle
        B = self.B
        self.active = np.zeros(B, dtype=bool)
        self.done = np.zeros(B, dtype=bool)
        self.success = np.zeros(B, dtype=bool)
        self.ctrl_step = np.zeros(B, dtype=np.int32)

        # FSM
        self.states = np.zeros(B, dtype=np.int32)
        self.state_enter_step = np.zeros(B, dtype=np.int32)
        self.state_reach_step = np.full(B, -1, dtype=np.int32)
        self._attempt_id = np.zeros(B, dtype=np.int64)

        # executor
        self.exec_pos = np.zeros((B, 3), dtype=np.float32)
        self.exec_quat = np.zeros((B, 4), dtype=np.float32)  # xyzw

        # rotation lock pos
        self.rot_lock_pos = np.zeros((B, 3), dtype=np.float32)

        # latches
        self.latched_start_pos = np.zeros((B, 3), dtype=np.float32)
        self.latched_start_quat = np.zeros((B, 4), dtype=np.float32)
        self.hold_quat = np.zeros((B, 4), dtype=np.float32)

        self.saved_count = 0
        self.attempted = 0
        self._last_log_t = time.perf_counter()
        self._last_action = np.zeros((B, 7), dtype=np.float32)

        # --- IO (shared): video/buffer/jsonl/pose fix ---
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
            num_envs=int(cfg.num_envs),
            cam_keys=self.cam_keys,
            save_video=bool(cfg.save_video),
            sample_every_steps=int(cfg.sample_every_steps),
            render_every_steps=int(cfg.render_every_steps),
            video_cfg=video_cfg,
            pose_fix=pose_fix,
            dt=0.02,
        )

    # -----------------------------
    # Episode control
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

        # init EE
        all_poses = self.env.robot.get_ee_pose(data)
        for idx in env_ids.tolist():
            pose = all_poses[idx]
            self.exec_pos[idx] = pose[:3]
            self.latched_start_pos[idx] = pose[:3]

            if len(pose) == 7:
                self.exec_quat[idx] = pose[3:7]
            elif len(pose) == 6:
                euler = pose[3:6]
                self.exec_quat[idx] = Rotation.from_euler("xyz", euler, degrees=False).as_quat().astype(np.float32)
            else:
                self.exec_quat[idx] = Rotation.identity().as_quat().astype(np.float32)

            self.exec_quat[idx] = normalize_quat(self.exec_quat[idx][None, :])[0].astype(np.float32)
            self.latched_start_quat[idx] = self.exec_quat[idx].copy()

        self.hold_quat[env_ids] = self.latched_start_quat[env_ids].copy()
        self.rot_lock_pos[env_ids] = self.latched_start_pos[env_ids].copy()

        # start state
        self.states[env_ids] = int(self.ST_APP1_LIFT_Z)

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
    # Core logic (unchanged FSM)
    # -----------------------------
    def _step_logic(self) -> None:
        cfg = self.cfg
        B = self.B

        data = self.env._state.data
        obs = self.env._state.obs

        running = self.active & (~self.done)
        if not np.any(running):
            return

        s = self.states

        # --- current object poses ---
        flower_pose = np.asarray(self.flower_body.get_pose(data), dtype=np.float32)
        flower_p = flower_pose[:, :3]

        vase_src_pose = np.asarray(self.vase_src_body.get_pose(data), dtype=np.float32)
        vase_dst_pose = np.asarray(self.vase_dst_body.get_pose(data), dtype=np.float32)
        vase_src_p = vase_src_pose[:, :3]
        vase_dst_p = vase_dst_pose[:, :3]

        g_off = np.array(cfg.grasp_offset, dtype=np.float32)

        # --- Base orientation (lock roll/pitch via base quat, yaw only) ---
        q_base = self.latched_start_quat  # (B,4) xyzw
        r_base = Rotation.from_quat(q_base)

        r_pick1 = Rotation.from_euler("z", float(cfg.pick1_yaw_deg), degrees=True)
        r_pick2 = Rotation.from_euler("z", float(cfg.pick2_yaw_deg), degrees=True)
        q_pick1 = normalize_quat((r_pick1 * r_base).as_quat().astype(np.float32))
        q_pick2 = normalize_quat((r_pick2 * r_base).as_quat().astype(np.float32))

        # yaw targets for robust reach-check
        yaw_tgt_pick1 = quat_to_yaw(q_pick1)
        yaw_tgt_pick2 = quat_to_yaw(q_pick2)

        def build_above(p_obj: np.ndarray, z: float) -> np.ndarray:
            out = p_obj.copy()
            out[:, 2] = float(z)
            return out

        # Safe height
        p_safe = obs["ee_pose"][:, :3].copy()
        p_safe[:, 2] = np.maximum(p_safe[:, 2], 0.20)

        # ---------------- pick1: coarse align ----------------
        p1_above = build_above(flower_p, cfg.lift_height_z)

        pick1_dx, pick1_dy = cfg.pick1_coarse_xy_offset
        p1_x = p1_above.copy()
        p1_x[:, 0] = flower_p[:, 0] + g_off[0] + float(pick1_dx)
        p1_y = p1_x.copy()
        p1_y[:, 1] = flower_p[:, 1] + g_off[1] + float(pick1_dy)

        p1_pre = p1_y.copy()
        p1_pre[:, 2] = flower_p[:, 2] + g_off[2] + float(cfg.pregrasp_z_margin)

        p1_fine_y = p1_pre.copy()
        p1_fine_y[:, 1] = flower_p[:, 1] + g_off[1]
        p1_fine_x = p1_fine_y.copy()
        p1_fine_x[:, 0] = flower_p[:, 0] + g_off[0]

        p1_grasp = p1_fine_x.copy()
        p1_grasp[:, 2] = flower_p[:, 2] + g_off[2]

        p1_lift = p1_grasp.copy()
        p1_lift[:, 2] = float(cfg.lift_height_z)

        # ---------------- place1 to dst ----------------
        dst_dx, dst_dy = cfg.place1_dst_hover_xy_offset
        p_dst_hover = p1_lift.copy()
        p_dst_hover[:, 0] = vase_dst_p[:, 0] + float(dst_dx)
        p_dst_hover[:, 1] = vase_dst_p[:, 1] + float(dst_dy)
        p_dst_hover[:, 2] = float(cfg.lift_height_z)

        p_dst_align = p_dst_hover.copy()
        p_dst_align[:, 2] += float(cfg.align_z_offset)

        p_dst_insert = p_dst_align.copy()
        p_dst_insert[:, 2] -= float(cfg.insert_depth)

        p_retreat1 = p_dst_align.copy()
        p_retreat1[:, 0] -= float(cfg.retreat_dx)

        # ---------------- pick2: recompute from current flower pose ----------------
        p2_above = build_above(flower_p, cfg.lift_height_z)

        p2_x = p2_above.copy()
        p2_x[:, 0] = flower_p[:, 0] + g_off[0]
        p2_y = p2_x.copy()
        p2_y[:, 1] = flower_p[:, 1] + g_off[1]

        p2_pre = p2_y.copy()
        p2_pre[:, 2] = flower_p[:, 2] + g_off[2] + float(cfg.pregrasp_z_margin)

        p2_fine_y = p2_pre.copy()
        p2_fine_y[:, 1] = flower_p[:, 1] + g_off[1]
        p2_fine_x = p2_fine_y.copy()
        p2_fine_x[:, 0] = flower_p[:, 0] + g_off[0]

        p2_grasp = p2_fine_x.copy()
        p2_grasp[:, 2] = flower_p[:, 2] + g_off[2] + float(cfg.pick2_grasp_z_extra)

        p2_lift = p2_grasp.copy()
        p2_lift[:, 2] = float(cfg.lift_height_z)

        # ---------------- place2 back to src ----------------
        src_dx, src_dy = cfg.place2_src_hover_xy_offset

        p_src_hover = p2_lift.copy()
        p_src_hover[:, 0] = vase_src_p[:, 0] + float(src_dx)
        p_src_hover[:, 1] = vase_src_p[:, 1] + float(src_dy)
        p_src_hover[:, 2] = float(cfg.lift_height_z)

        p_src_align = p_src_hover.copy()
        p_src_align[:, 2] += float(cfg.align_z_offset)

        p_src_insert = p_src_align.copy()
        p_src_insert[:, 2] -= float(cfg.insert_depth)
        p_src_insert[:, 2] += float(cfg.place2_insert_z_extra)

        p_retreat2 = p_src_align.copy()
        p_retreat2[:, 0] -= float(cfg.retreat_dx)

        # --- target assignment ---
        tgt_pos = self.exec_pos.copy()
        grip_cmd = np.full((B,), float(cfg.gripper_open), dtype=np.float32)

        q_hold = normalize_quat(self.hold_quat.copy()).astype(np.float32)

        def set_target(state_id: int, pos: np.ndarray, grip: float):
            mask = running & (s == state_id)
            if np.any(mask):
                tgt_pos[mask] = pos[mask]
                grip_cmd[mask] = float(grip)

        # pick1
        set_target(self.ST_APP1_LIFT_Z, p_safe, cfg.gripper_open)
        set_target(self.ST_APP1_ALIGN_X, p1_x, cfg.gripper_open)
        set_target(self.ST_APP1_ALIGN_Y, p1_y, cfg.gripper_open)
        set_target(self.ST_PRE1_Z, p1_pre, cfg.gripper_open)
        set_target(self.ST_PRE1_ALIGN_Y, p1_fine_y, cfg.gripper_open)
        set_target(self.ST_PRE1_ALIGN_X, p1_fine_x, cfg.gripper_open)
        set_target(self.ST_DESCEND1, p1_grasp, cfg.gripper_open)
        set_target(self.ST_CLOSE1, p1_grasp, cfg.gripper_close)
        set_target(self.ST_LIFT1, p1_lift, cfg.gripper_close)

        # place1
        set_target(self.ST_TRP1_ALIGN_X, p_dst_hover, cfg.gripper_close)
        set_target(self.ST_TRP1_ALIGN_Y, p_dst_hover, cfg.gripper_close)
        set_target(self.ST_PLACE1_DESCEND, p_dst_insert, cfg.gripper_close)
        set_target(self.ST_OPEN1, p_dst_insert, cfg.gripper_open)
        set_target(self.ST_RETREAT1, p_retreat1, cfg.gripper_open)

        # pick2
        set_target(self.ST_APP2_LIFT_Z, p_safe, cfg.gripper_open)
        set_target(self.ST_APP2_ALIGN_X, p2_x, cfg.gripper_open)
        set_target(self.ST_APP2_ALIGN_Y, p2_y, cfg.gripper_open)
        set_target(self.ST_PRE2_Z, p2_pre, cfg.gripper_open)
        set_target(self.ST_PRE2_ALIGN_Y, p2_fine_y, cfg.gripper_open)
        set_target(self.ST_PRE2_ALIGN_X, p2_fine_x, cfg.gripper_open)
        set_target(self.ST_DESCEND2, p2_grasp, cfg.gripper_open)
        set_target(self.ST_CLOSE2, p2_grasp, cfg.gripper_close)
        set_target(self.ST_LIFT2, p2_lift, cfg.gripper_close)

        # place2
        set_target(self.ST_TRP2_ALIGN_X, p_src_hover, cfg.gripper_close)
        set_target(self.ST_TRP2_ALIGN_Y, p_src_hover, cfg.gripper_close)
        set_target(self.ST_PLACE2_DESCEND, p_src_insert, cfg.gripper_close)
        set_target(self.ST_OPEN2, p_src_insert, cfg.gripper_open)
        set_target(self.ST_RETREAT2, p_retreat2, cfg.gripper_open)

        # --- execute position ---
        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos, float(cfg.max_dp))

        # --- execute rotation: yaw-only stepping in rotation states ---
        rot_states = np.array([self.ST_ROT_PICK1, self.ST_ROT_PICK2], dtype=np.int32)
        rot_mask = running & np.isin(s, rot_states)

        if np.any(rot_mask):
            ee_pose = obs["ee_pose"]
            if ee_pose is not None and ee_pose.shape[1] == 7:
                ee_quat = ee_pose[:, 3:7].astype(np.float32)
            elif ee_pose is not None and ee_pose.shape[1] == 6:
                ee_quat = Rotation.from_euler("xyz", ee_pose[:, 3:6], degrees=False).as_quat().astype(np.float32)
            else:
                ee_quat = ref_quat.copy()
            curr_yaw = quat_to_yaw(ee_quat)

            yaw_tgt = curr_yaw.copy()
            yaw_tgt[s == self.ST_ROT_PICK1] = yaw_tgt_pick1[s == self.ST_ROT_PICK1]
            yaw_tgt[s == self.ST_ROT_PICK2] = yaw_tgt_pick2[s == self.ST_ROT_PICK2]

            dyaw = wrap_to_pi(yaw_tgt - curr_yaw)
            dyaw_step = np.clip(dyaw, -float(cfg.max_dr), float(cfg.max_dr))

            base_rpy = Rotation.from_quat(self.latched_start_quat).as_euler("xyz", degrees=False).astype(np.float32)
            new_rpy = base_rpy.copy()
            new_rpy[:, 2] = (curr_yaw + dyaw_step).astype(np.float32)
            new_q = Rotation.from_euler("xyz", new_rpy, degrees=False).as_quat().astype(np.float32)

            self.exec_quat[rot_mask] = normalize_quat(new_q[rot_mask])
        # else: keep quat unchanged

        self.exec_quat = normalize_quat(self.exec_quat).astype(np.float32)

        # --- action (eef_relative) ---
        ref_pose_6d = self.env.robot.ref_ee_pose
        ref_pos = ref_pose_6d[:, :3]
        ref_euler = ref_pose_6d[:, 3:6]
        ref_quat = Rotation.from_euler("xyz", ref_euler, degrees=False).as_quat().astype(np.float32)

        action = np.zeros((B, 7), dtype=np.float32)
        action[:, :3] = (self.exec_pos - ref_pos) * float(cfg.action_pos_scale)

        rotvec_cmd = np.zeros((B, 3), dtype=np.float32)
        if np.any(rot_mask):
            idxs = np.where(rot_mask)[0]
            for i in idxs:
                r_des = Rotation.from_quat(self.exec_quat[i])
                r_ref = Rotation.from_quat(ref_quat[i])
                r_e = r_des * r_ref.inv()
                rv = r_e.as_rotvec().astype(np.float32)
                mag = float(np.linalg.norm(rv)) + 1e-9
                scale = min(1.0, float(cfg.max_dr) / mag)
                rotvec_cmd[i] = rv * scale * float(cfg.rot_gain)

        action[:, 3:6] = rotvec_cmd
        action[:, 6] = grip_cmd * float(cfg.action_grip_scale)

        self._last_action[:] = action
        self.env.step(action)

        # refresh obs after step for checks
        obs = self.env._state.obs
        print("state",s)
        # ---------------- transitions ----------------
        def is_pos_reached(target: np.ndarray) -> np.ndarray:
            return np.linalg.norm(self.exec_pos - target, axis=1) < float(cfg.pos_tol)

        def rot_state_done(mask_in_state: np.ndarray, yaw_target: np.ndarray) -> np.ndarray:
            ee_pose = obs["ee_pose"]
            if ee_pose is not None and ee_pose.shape[1] == 7:
                ee_quat = ee_pose[:, 3:7].astype(np.float32)
            elif ee_pose is not None and ee_pose.shape[1] == 6:
                ee_quat = Rotation.from_euler("xyz", ee_pose[:, 3:6], degrees=False).as_quat().astype(np.float32)
            else:
                ee_quat = ref_quat.copy()
            curr_yaw = Rotation.from_quat(ee_quat).as_euler("xyz", degrees=False)[:, 2].astype(np.float32)

            dyaw = wrap_to_pi(curr_yaw - yaw_target)
            ok = np.abs(dyaw) < float(cfg.yaw_tol)

            just = mask_in_state & ok & (self.state_reach_step == -1)
            if np.any(just):
                self.state_reach_step[just] = self.ctrl_step[just]
            dwell = (self.ctrl_step - self.state_reach_step) >= int(cfg.rot_dwell_steps)

            too_long = (self.ctrl_step - self.state_enter_step) >= int(cfg.rot_state_timeout_steps)
            return mask_in_state & (((self.state_reach_step != -1) & dwell & ok) | too_long)

        def _check_and_dwell(state_id: int, pos_tgt: np.ndarray) -> np.ndarray:
            in_state = running & (s == state_id)
            ok = is_pos_reached(pos_tgt)
            just = in_state & ok & (self.state_reach_step == -1)
            if np.any(just):
                self.state_reach_step[just] = self.ctrl_step[just]
            dwell = (self.ctrl_step - self.state_reach_step) >= int(cfg.waypoint_dwell_steps)
            return in_state & (self.state_reach_step != -1) & dwell & ok

        # pick1
        self._enter_state(_check_and_dwell(self.ST_APP1_LIFT_Z, p_safe), self.ST_APP1_ALIGN_X)
        self._enter_state(_check_and_dwell(self.ST_APP1_ALIGN_X, p1_x), self.ST_APP1_ALIGN_Y)

        to_rot1 = _check_and_dwell(self.ST_APP1_ALIGN_Y, p1_y)
        if np.any(to_rot1):
            self.rot_lock_pos[to_rot1] = self.exec_pos[to_rot1].copy()
            self._enter_state(to_rot1, self.ST_ROT_PICK1)

        done_rot1 = rot_state_done(running & (s == self.ST_ROT_PICK1), yaw_tgt_pick1)
        if np.any(done_rot1):
            self.hold_quat[done_rot1] = q_pick1[done_rot1].copy()
            self._enter_state(done_rot1, self.ST_PRE1_Z)

        self._enter_state(_check_and_dwell(self.ST_PRE1_Z, p1_pre), self.ST_PRE1_ALIGN_Y)
        self._enter_state(_check_and_dwell(self.ST_PRE1_ALIGN_Y, p1_fine_y), self.ST_PRE1_ALIGN_X)
        self._enter_state(_check_and_dwell(self.ST_PRE1_ALIGN_X, p1_fine_x), self.ST_DESCEND1)
        self._enter_state(_check_and_dwell(self.ST_DESCEND1, p1_grasp), self.ST_CLOSE1)

        mask_close1 = running & (s == self.ST_CLOSE1)
        if np.any(mask_close1):
            t_in = self.ctrl_step - self.state_enter_step
            done_close = t_in >= int(cfg.close_hold_steps)
            self._enter_state(mask_close1 & done_close, self.ST_LIFT1)

        self._enter_state(_check_and_dwell(self.ST_LIFT1, p1_lift), self.ST_TRP1_ALIGN_X)

        # place1
        self._enter_state(_check_and_dwell(self.ST_TRP1_ALIGN_X, p_dst_hover), self.ST_TRP1_ALIGN_Y)
        self._enter_state(_check_and_dwell(self.ST_TRP1_ALIGN_Y, p_dst_hover), self.ST_PLACE1_DESCEND)
        self._enter_state(_check_and_dwell(self.ST_PLACE1_DESCEND, p_dst_insert), self.ST_OPEN1)

        mask_open1 = running & (s == self.ST_OPEN1)
        if np.any(mask_open1):
            t_in = self.ctrl_step - self.state_enter_step
            done_open = t_in >= int(cfg.close_hold_steps)
            self._enter_state(mask_open1 & done_open, self.ST_RETREAT1)

        self._enter_state(_check_and_dwell(self.ST_RETREAT1, p_retreat1), self.ST_APP2_LIFT_Z)

        # pick2
        self._enter_state(_check_and_dwell(self.ST_APP2_LIFT_Z, p_safe), self.ST_APP2_ALIGN_X)
        self._enter_state(_check_and_dwell(self.ST_APP2_ALIGN_X, p2_x), self.ST_APP2_ALIGN_Y)

        to_rot2 = _check_and_dwell(self.ST_APP2_ALIGN_Y, p2_y)
        if np.any(to_rot2):
            self.rot_lock_pos[to_rot2] = self.exec_pos[to_rot2].copy()
            self._enter_state(to_rot2, self.ST_ROT_PICK2)

        done_rot2 = rot_state_done(running & (s == self.ST_ROT_PICK2), yaw_tgt_pick2)
        if np.any(done_rot2):
            self.hold_quat[done_rot2] = q_pick2[done_rot2].copy()
            self._enter_state(done_rot2, self.ST_PRE2_Z)

        self._enter_state(_check_and_dwell(self.ST_PRE2_Z, p2_pre), self.ST_PRE2_ALIGN_Y)
        self._enter_state(_check_and_dwell(self.ST_PRE2_ALIGN_Y, p2_fine_y), self.ST_PRE2_ALIGN_X)
        self._enter_state(_check_and_dwell(self.ST_PRE2_ALIGN_X, p2_fine_x), self.ST_DESCEND2)
        self._enter_state(_check_and_dwell(self.ST_DESCEND2, p2_grasp), self.ST_CLOSE2)

        mask_close2 = running & (s == self.ST_CLOSE2)
        if np.any(mask_close2):
            t_in = self.ctrl_step - self.state_enter_step
            done_close = t_in >= int(cfg.close_hold_steps)
            self._enter_state(mask_close2 & done_close, self.ST_LIFT2)

        self._enter_state(_check_and_dwell(self.ST_LIFT2, p2_lift), self.ST_TRP2_ALIGN_X)

        # place2
        self._enter_state(_check_and_dwell(self.ST_TRP2_ALIGN_X, p_src_hover), self.ST_TRP2_ALIGN_Y)
        self._enter_state(_check_and_dwell(self.ST_TRP2_ALIGN_Y, p_src_hover), self.ST_PLACE2_DESCEND)
        self._enter_state(_check_and_dwell(self.ST_PLACE2_DESCEND, p_src_insert), self.ST_OPEN2)

        mask_open2 = running & (s == self.ST_OPEN2)
        if np.any(mask_open2):
            t_in = self.ctrl_step - self.state_enter_step
            done_open = t_in >= int(cfg.close_hold_steps)
            self._enter_state(mask_open2 & done_open, self.ST_RETREAT2)

        self._enter_state(_check_and_dwell(self.ST_RETREAT2, p_retreat2), self.ST_DONE)

    # -----------------------------
    # Collection loop
    # -----------------------------
    def collect(self) -> None:
        cfg = self.cfg
        target_n = int(cfg.data_size)

        all_ids = np.arange(self.B, dtype=np.int64)
        self.start_episodes(all_ids, seed=int(cfg.seed))

        print(f"Starting ArrangeFlowers Collection (refactored). Target: {target_n}")

        while self.saved_count < target_n:
            self._step_logic()

            running = self.active & (~self.done)
            obs = self.env._state.obs

            # capture (before render) so frame_idx aligns with video frame count
            sample_mask = running & ((self.ctrl_step % int(cfg.sample_every_steps)) == 0)
            for env_id in np.where(sample_mask)[0].tolist():
                self.io.capture_step(
                    env_id=int(env_id),
                    ctrl_step=int(self.ctrl_step[env_id]),
                    state=int(self.states[env_id]),
                    obs=obs,
                    last_action=self._last_action,
                    success=bool(self.success[env_id]),
                    reward=float(self.env._state.reward[env_id]) if hasattr(self.env._state, "reward") else 0.0,
                    extra_step=None,
                )

            if cfg.save_video:
                render_mask = running & ((self.ctrl_step % int(cfg.render_every_steps)) == 0)
                for env_id in np.where(render_mask)[0].tolist():
                    self.io.maybe_write_video(env_id=int(env_id), obs=obs)
                    
            self.ctrl_step[running] += 1

            # termination conditions
            for i in range(self.B):
                if not running[i]:
                    continue
                fsm_done = (int(self.states[i]) == int(self.ST_DONE))
                timeout = (int(self.ctrl_step[i]) >= int(cfg.max_ctrl_steps))
                env_success = bool(getattr(self.env, "success_latched", np.zeros((self.B,), dtype=bool))[i])
                if fsm_done or timeout or env_success:
                    self.done[i] = True
                    self.success[i] = bool(env_success) or bool(fsm_done)

            # finalize & restart
            for i in range(self.B):
                if not (self.active[i] and self.done[i]):
                    continue

                prompt = str(cfg.instruction)
                extra_ep = {
                    "task_name": "arrange_flowers",
                    "attempt_id": int(self._attempt_id[i]),
                }

                if bool(self.success[i]) and (self.saved_count < target_n):
                    ep_idx = int(self.saved_count)
                    saved = self.io.finalize_episode(
                        env_id=int(i),
                        ep_idx=ep_idx,
                        success=True,
                        prompt=prompt,
                        extra_episode=extra_ep,
                    )
                    
                    self.saved_count += 1
                else:
                    self.io.finalize_episode(
                        env_id=int(i),
                        ep_idx=int(self.saved_count),
                        success=False,
                        prompt=prompt,
                        extra_episode=extra_ep,
                    )

                if self.saved_count < target_n:
                    self.active[i] = False
                    new_seed = int(cfg.seed + self.attempted)
                    self.start_episodes(np.array([i], dtype=np.int64), seed=new_seed)
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


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--data_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no_video", action="store_true")

    p.add_argument("--pick1_yaw_deg", type=float, default=None)
    p.add_argument("--pick2_yaw_deg", type=float, default=None)
    p.add_argument("--pregrasp_z_margin", type=float, default=None)

    # offsets knobs
    p.add_argument("--pick1_coarse_dx", type=float, default=None)
    p.add_argument("--pick1_coarse_dy", type=float, default=None)
    p.add_argument("--dst_hover_dx", type=float, default=None)
    p.add_argument("--dst_hover_dy", type=float, default=None)
    p.add_argument("--pick2_grasp_z_extra", type=float, default=None)
    p.add_argument("--src_hover_dx", type=float, default=None)
    p.add_argument("--src_hover_dy", type=float, default=None)
    p.add_argument("--src_insert_z_extra", type=float, default=None)

    # action scaling
    p.add_argument("--action_pos_scale", type=float, default=None)
    p.add_argument("--action_grip_scale", type=float, default=None)

    # MotrixSim toggles
    p.add_argument("--motrix_video", action="store_true", help="Enable MotrixSim renderer video recording.")

    p.add_argument("--motrix_video_fps", type=int, default=None)
    p.add_argument("--motrix_video_width", type=int, default=None)
    p.add_argument("--motrix_video_height", type=int, default=None)

    args = p.parse_args()

    cfg = CollectorCfg(
        save_dir=args.save_dir if args.save_dir else CollectorCfg.save_dir,
        num_envs=args.num_envs if args.num_envs else CollectorCfg.num_envs,
        data_size=args.data_size if args.data_size is not None else CollectorCfg.data_size,
        seed=args.seed if args.seed is not None else CollectorCfg.seed,
        save_video=(not args.no_video),

        pick1_yaw_deg=(args.pick1_yaw_deg if args.pick1_yaw_deg is not None else CollectorCfg.pick1_yaw_deg),
        pick2_yaw_deg=(args.pick2_yaw_deg if args.pick2_yaw_deg is not None else CollectorCfg.pick2_yaw_deg),
        pregrasp_z_margin=(args.pregrasp_z_margin if args.pregrasp_z_margin is not None else CollectorCfg.pregrasp_z_margin),

        pick1_coarse_xy_offset=(
            args.pick1_coarse_dx if args.pick1_coarse_dx is not None else CollectorCfg.pick1_coarse_xy_offset[0],
            args.pick1_coarse_dy if args.pick1_coarse_dy is not None else CollectorCfg.pick1_coarse_xy_offset[1],
        ),
        place1_dst_hover_xy_offset=(
            args.dst_hover_dx if args.dst_hover_dx is not None else CollectorCfg.place1_dst_hover_xy_offset[0],
            args.dst_hover_dy if args.dst_hover_dy is not None else CollectorCfg.place1_dst_hover_xy_offset[1],
        ),
        pick2_grasp_z_extra=(
            args.pick2_grasp_z_extra if args.pick2_grasp_z_extra is not None else CollectorCfg.pick2_grasp_z_extra
        ),
        place2_src_hover_xy_offset=(
            args.src_hover_dx if args.src_hover_dx is not None else CollectorCfg.place2_src_hover_xy_offset[0],
            args.src_hover_dy if args.src_hover_dy is not None else CollectorCfg.place2_src_hover_xy_offset[1],
        ),
        place2_insert_z_extra=(
            args.src_insert_z_extra if args.src_insert_z_extra is not None else CollectorCfg.place2_insert_z_extra
        ),

        action_pos_scale=(args.action_pos_scale if args.action_pos_scale is not None else CollectorCfg.action_pos_scale),
        action_grip_scale=(args.action_grip_scale if args.action_grip_scale is not None else CollectorCfg.action_grip_scale),

        enable_motrix_video = bool(args.motrix_video),
        motrix_video_fps=(args.motrix_video_fps if args.motrix_video_fps is not None else CollectorCfg.motrix_video_fps),
        motrix_video_width=(
            args.motrix_video_width if args.motrix_video_width is not None else CollectorCfg.motrix_video_width
        ),
        motrix_video_height=(
            args.motrix_video_height if args.motrix_video_height is not None else CollectorCfg.motrix_video_height
        ),
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