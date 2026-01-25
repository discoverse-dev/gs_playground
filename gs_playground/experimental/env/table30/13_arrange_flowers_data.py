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


def normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / (n + 1e-9)


# -----------------------------------------------------------------------------
# Video Writer (3DGS)
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
    data_size: int = 1
    num_envs: int = 1
    seed: int = 0
    save_dir: str = "./data/table30_arrange_flowers_vase_swap"

    # env control
    max_ctrl_steps: int = 1700

    # motion params
    max_dp: float = 0.005
    pos_tol: float = 0.002

    # rotation control
    rot_gain: float = 1.0
    max_dr: float = 0.01
    angle_tol: float = 0.08

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
    rot_dwell_steps: int = 15

    # retreat
    retreat_dx: float = 0.08

    # yaw interfaces (degrees) - default 0 but adjustable
    pick1_yaw_deg: float = -30.0
    pick2_yaw_deg: float = -30.0

    # sampling / render (3DGS)
    sample_every_steps: int = 1
    save_video: bool = True
    render_every_steps: int = 1
    video_fps: int = 30
    video_w: int = 640
    video_h: int = 480
    cam_view_key: Optional[str] = "pixels/view_0"

    # -----------------------------
    # MotrixSim renderer video (DEBUG)
    # -----------------------------
    enable_motrix_video: bool = True
    motrix_video_fps: int = 30
    motrix_video_width: int = 640
    motrix_video_height: int = 480
    # output dir will be forced to {save_dir}/videos

    instruction: str = (
        "Pick up the flower from the source vase, place it into the target vase, "
        "then pick it up again and place it back into the source vase. "
        "Each motion first aligns XY, then performs Z motion. Rotation only changes yaw. "
        "Before grasp, do a pregrasp descend then fine-align (Z->Y->X), then final descend and grasp."
    )


# -----------------------------------------------------------------------------
# Collector
# -----------------------------------------------------------------------------
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
        self.videos_dir = os.path.join(cfg.save_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

        # Env cfg
        self.env_cfg = env_cfg if env_cfg is not None else ArrangeFlowersEnvCfg()

        # -----------------------------
        # Pass MotrixSim recorder config into env cfg (best-effort, like fruits collector)
        # -----------------------------
        setattr(self.env_cfg, "enable_motrix_video", bool(cfg.enable_motrix_video))
        setattr(self.env_cfg, "motrix_video_fps", int(cfg.motrix_video_fps))
        setattr(self.env_cfg, "motrix_video_width", int(cfg.motrix_video_width))
        setattr(self.env_cfg, "motrix_video_height", int(cfg.motrix_video_height))
        setattr(self.env_cfg, "motrix_video_output_dir", self.videos_dir)

        self.env = ArrangeFlowersEnv(self.env_cfg, num_envs=int(cfg.num_envs))
        self.env.reset()

        self.model = self.env.model
        self.B = int(cfg.num_envs)

        self.flower_body = self.env.flower_body
        self.vase_src_body = getattr(self.env, "vase_src_body", self.env.vase_body)
        self.vase_dst_body = getattr(self.env, "vase_dst_body", None)
        if self.vase_dst_body is None:
            raise RuntimeError("Env missing vase_dst_body (vase2). Please update env to include vase2.")

        self.cam_view_key = cfg.cam_view_key or "pixels/view_0"

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

        # executor
        self.exec_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.exec_quat = np.zeros((self.B, 4), dtype=np.float32)

        # rotation lock pos (critical)
        self.rot_lock_pos = np.zeros((self.B, 3), dtype=np.float32)

        # latches
        self.latched_start_pos = np.zeros((self.B, 3), dtype=np.float32)
        self.latched_start_quat = np.zeros((self.B, 4), dtype=np.float32)
        self.hold_quat = np.zeros((self.B, 4), dtype=np.float32)

        # buffers + 3DGS video
        self.buffers: List[Dict[str, Any]] = [self._new_buffer() for _ in range(self.B)]
        self.video_writers: List[Optional[EpisodeVideoWriter]] = [None] * self.B
        self._tmp_video_paths: List[str] = [os.path.join(self.videos_dir, f"_tmp_env{i}.mp4") for i in range(self.B)]

        self.saved_count = 0
        self.attempted = 0
        self._last_log_t = time.perf_counter()
        self._last_action = np.zeros((self.B, 7), dtype=np.float32)

        # motrix recorder sanity flag
        self._warned_no_motrix = False

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
        for idx in env_ids:
            pose = all_poses[idx]
            self.exec_pos[idx] = pose[:3]
            self.latched_start_pos[idx] = pose[:3]
            if len(pose) == 7:
                self.exec_quat[idx] = pose[3:]
            elif len(pose) == 6:
                euler = pose[3:]
                self.exec_quat[idx] = Rotation.from_euler("xyz", euler, degrees=False).as_quat()
            else:
                self.exec_quat[idx] = Rotation.identity().as_quat()

            self.exec_quat[idx] = normalize_quat(self.exec_quat[idx][None, :])[0].astype(np.float32)
            self.latched_start_quat[idx] = self.exec_quat[idx].copy()

        self.hold_quat[env_ids] = self.latched_start_quat[env_ids].copy()
        self.rot_lock_pos[env_ids] = self.latched_start_pos[env_ids].copy()

        # start state
        self.states[env_ids] = self.ST_APP1_LIFT_Z

        for env_id in env_ids.tolist():
            self.buffers[env_id] = self._new_buffer()

            if self.video_writers[env_id] is not None:
                self.video_writers[env_id].close()
                self.video_writers[env_id] = None

            if self.cfg.save_video:
                self._reset_video_writer(env_id)

            # -----------------------------
            # Restart MotrixSim recorder for env_id==0 (Motrix typically records only first env)
            # -----------------------------
            if env_id == 0 and bool(self.cfg.enable_motrix_video):
                rec = getattr(self.env, "_motrix_recorder", None)
                if rec is not None:
                    ep_idx = int(self.saved_count)
                    rec.restart_episode(self.videos_dir, ep_idx)
                else:
                    if not self._warned_no_motrix:
                        print("[WARN] enable_motrix_video=True but env._motrix_recorder is None. "
                              "Please implement Motrix recorder in ArrangeFlowersEnv like ArrangeFruitsEnv.")
                        self._warned_no_motrix = True

            self._attempt_id[env_id] += 1
            self.attempted += 1

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

    def _flush_jsonl(self, env_id: int, ep_idx: int, vid_path: str):
        path = os.path.join(self.cfg.save_dir, f"episode_{ep_idx:05d}.jsonl")
        buf = self.buffers[env_id]
        n = len(buf["times"])
        prompt = self.cfg.instruction

        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                rec = {
                    "images_1": {"url": vid_path, "type": "video", "frame_idx": i},
                    "prompt": prompt,
                    "qpos": buf["qpos"][i],
                    "ee_pose": buf["ee_pose"][i],
                    "gripper": buf["gripper"][i],
                    "ctrl": buf["ctrl"][i],
                    "is_robot": True,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _finalize_episode(self, env_id: int) -> None:
        if self.video_writers[env_id]:
            self.video_writers[env_id].close()
            self.video_writers[env_id] = None

        if self.saved_count < self.cfg.data_size:
            ep_idx = int(self.saved_count)
            final_video_path = f"videos/episode_{ep_idx:05d}.mp4"
            abs_video_path = os.path.join(self.cfg.save_dir, final_video_path)

            if self.cfg.save_video and os.path.exists(self._tmp_video_paths[env_id]):
                shutil.move(self._tmp_video_paths[env_id], abs_video_path)

            self._flush_jsonl(env_id, ep_idx, final_video_path)
            self.saved_count += 1
            print(f"[Saved] Episode {ep_idx}. Total saved: {self.saved_count}")

        if os.path.exists(self._tmp_video_paths[env_id]):
            try:
                os.remove(self._tmp_video_paths[env_id])
            except Exception:
                pass
        self.buffers[env_id] = self._new_buffer()

    def _enter_state(self, mask: np.ndarray, new_state: int) -> None:
        if not np.any(mask):
            return
        self.states[mask] = new_state
        self.state_enter_step[mask] = self.ctrl_step[mask].copy()
        self.state_reach_step[mask] = -1

    def _step_logic(self) -> None:
        cfg = self.cfg
        B = self.B

        # NOTE: obs will be refreshed after env.step for transition checks
        obs = self.env._state.obs
        data = self.env._state.data

        running = self.active & (~self.done)
        if not np.any(running):
            return

        s = self.states
        print("state",s)

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

        def build_above(p_obj: np.ndarray, z: float) -> np.ndarray:
            out = p_obj.copy()
            out[:, 2] = float(z)
            return out

        # Safe height
        p_safe = obs["ee_pose"][:, :3].copy()
        p_safe[:, 2] = np.maximum(p_safe[:, 2], 0.20)

        # ---------------- pick1: coarse align ----------------
        p1_above = build_above(flower_p, cfg.lift_height_z)
        p1_x = p1_above.copy()
        p1_x[:, 0] = flower_p[:, 0] + g_off[0] - 0.03
        p1_y = p1_x.copy()
        p1_y[:, 1] = flower_p[:, 1] + g_off[1]

        p1_grasp = p1_y.copy()
        p1_grasp[:, 2] = flower_p[:, 2] + g_off[2]

        p1_pre = p1_y.copy()
        p1_pre[:, 2] = flower_p[:, 2] + g_off[2] + float(cfg.pregrasp_z_margin)

        p1_fine_y = p1_pre.copy()
        p1_fine_y[:, 1] = flower_p[:, 1] + g_off[1]
        p1_fine_x = p1_fine_y.copy()
        p1_fine_x[:, 0] = flower_p[:, 0] + g_off[0]
        p1_grasp = p1_fine_x
        p1_lift = p1_grasp.copy()
        p1_lift[:, 2] = float(cfg.lift_height_z)

        # ---------------- place1 to dst ----------------
        p_dst_hover = p1_lift.copy()
        p_dst_hover[:, 0] = vase_dst_p[:, 0] + 0.03
        p_dst_hover[:, 1] = vase_dst_p[:, 1] + 0.02
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

        p2_grasp = p2_y.copy()
        p2_grasp[:, 2] = flower_p[:, 2] + g_off[2] + 0.05

        p2_pre = p2_y.copy()
        p2_pre[:, 2] = flower_p[:, 2] + g_off[2] + float(cfg.pregrasp_z_margin)

        p2_fine_y = p2_pre.copy()
        p2_fine_y[:, 1] = flower_p[:, 1] + g_off[1]
        p2_fine_x = p2_fine_y.copy()
        p2_fine_x[:, 0] = flower_p[:, 0] + g_off[0]

        p2_lift = p2_grasp.copy()
        p2_lift[:, 2] = float(cfg.lift_height_z)

        # ---------------- place2 back to src ----------------
        p_src_hover = p2_lift.copy()
        p_src_hover[:, 0] = vase_src_p[:, 0]+0.05
        p_src_hover[:, 1] = vase_src_p[:, 1]+0.03
        p_src_hover[:, 2] = float(cfg.lift_height_z)

        p_src_align = p_src_hover.copy()
        p_src_align[:, 2] += float(cfg.align_z_offset)

        p_src_insert = p_src_align.copy()
        p_src_insert[:, 2] -= float(cfg.insert_depth)
        p_src_insert[:, 2] -= 0.08
        p_retreat2 = p_src_align.copy()
        p_retreat2[:, 0] -= float(cfg.retreat_dx)

        # --- target assignment ---
        tgt_pos_curr = self.exec_pos.copy()
        tgt_quat_curr = self.exec_quat.copy()
        grip_cmd = np.full((B,), cfg.gripper_open, dtype=np.float32)

        def set_target(state_id: int, pos: np.ndarray, quat: np.ndarray, grip: float):
            mask = running & (s == state_id)
            if np.any(mask):
                tgt_pos_curr[mask] = pos[mask]
                tgt_quat_curr[mask] = quat[mask]
                grip_cmd[mask] = float(grip)

        q_hold = normalize_quat(self.hold_quat.copy()).astype(np.float32)

        # pick1
        set_target(self.ST_APP1_LIFT_Z, p_safe, q_hold, cfg.gripper_open)
        set_target(self.ST_APP1_ALIGN_X, p1_x, q_hold, cfg.gripper_open)
        set_target(self.ST_APP1_ALIGN_Y, p1_y, q_hold, cfg.gripper_open)
        set_target(self.ST_ROT_PICK1, self.rot_lock_pos, q_pick1, cfg.gripper_open)
        set_target(self.ST_PRE1_Z, p1_pre, q_pick1, cfg.gripper_open)
        set_target(self.ST_PRE1_ALIGN_Y, p1_fine_y, q_pick1, cfg.gripper_open)
        set_target(self.ST_PRE1_ALIGN_X, p1_fine_x, q_pick1, cfg.gripper_open)
        set_target(self.ST_DESCEND1, p1_grasp, q_pick1, cfg.gripper_open)
        set_target(self.ST_CLOSE1, p1_grasp, q_pick1, cfg.gripper_close)
        set_target(self.ST_LIFT1, p1_lift, q_pick1, cfg.gripper_close)

        # place1
        set_target(self.ST_TRP1_ALIGN_X, p_dst_hover, q_hold, cfg.gripper_close)
        set_target(self.ST_TRP1_ALIGN_Y, p_dst_hover, q_hold, cfg.gripper_close)
        set_target(self.ST_PLACE1_DESCEND, p_dst_insert, q_hold, cfg.gripper_close)
        set_target(self.ST_OPEN1, p_dst_insert, q_hold, cfg.gripper_open)
        set_target(self.ST_RETREAT1, p_retreat1, q_hold, cfg.gripper_open)

        # pick2
        set_target(self.ST_APP2_LIFT_Z, p_safe, q_hold, cfg.gripper_open)
        set_target(self.ST_APP2_ALIGN_X, p2_x, q_hold, cfg.gripper_open)
        set_target(self.ST_APP2_ALIGN_Y, p2_y, q_hold, cfg.gripper_open)
        set_target(self.ST_ROT_PICK2, self.rot_lock_pos, q_pick2, cfg.gripper_open)
        set_target(self.ST_PRE2_Z, p2_pre, q_pick2, cfg.gripper_open)
        set_target(self.ST_PRE2_ALIGN_Y, p2_fine_y, q_pick2, cfg.gripper_open)
        set_target(self.ST_PRE2_ALIGN_X, p2_fine_x, q_pick2, cfg.gripper_open)
        set_target(self.ST_DESCEND2, p2_grasp, q_pick2, cfg.gripper_open)
        set_target(self.ST_CLOSE2, p2_grasp, q_pick2, cfg.gripper_close)
        set_target(self.ST_LIFT2, p2_lift, q_pick2, cfg.gripper_close)

        # place2
        set_target(self.ST_TRP2_ALIGN_X, p_src_hover, q_hold, cfg.gripper_close)
        set_target(self.ST_TRP2_ALIGN_Y, p_src_hover, q_hold, cfg.gripper_close)
        set_target(self.ST_PLACE2_DESCEND, p_src_insert, q_hold, cfg.gripper_close)
        set_target(self.ST_OPEN2, p_src_insert, q_hold, cfg.gripper_open)
        set_target(self.ST_RETREAT2, p_retreat2, q_hold, cfg.gripper_open)

        # --- execute ---
        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos_curr, cfg.max_dp)

        # incremental rotation (only apply control command during rotation states)
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
        self.exec_quat = normalize_quat(self.exec_quat).astype(np.float32)

        # action (eef_relative)
        ref_pose_6d = self.env.robot.ref_ee_pose
        ref_pos = ref_pose_6d[:, :3]
        ref_euler = ref_pose_6d[:, 3:6]
        ref_quat = Rotation.from_euler("xyz", ref_euler, degrees=False).as_quat().astype(np.float32)

        pos_err = (self.exec_pos - ref_pos) * 0.5

        rot_states = np.array([self.ST_ROT_PICK1, self.ST_ROT_PICK2], dtype=np.int32)
        rot_mask = running & np.isin(s, rot_states)

        rotvec_cmd = np.zeros((B, 3), dtype=np.float32)
        if np.any(rot_mask):
            idxs = np.where(rot_mask)[0]
            for i in idxs:
                r_des = Rotation.from_quat(self.exec_quat[i])
                r_ref = Rotation.from_quat(ref_quat[i])
                r_e = r_des * r_ref.inv()
                rv = r_e.as_rotvec()
                mag = np.linalg.norm(rv) + 1e-9
                scale = np.minimum(1.0, float(cfg.max_dr) / mag)
                rotvec_cmd[i] = rv * scale * float(cfg.rot_gain)

        action = np.zeros((B, 7), dtype=np.float32)
        action[:, :3] = pos_err * 0.5
        action[:, 3:6] = rotvec_cmd
        action[:, 6] = grip_cmd *0.98

        self._last_action[:] = action
        self.env.step(action)

        # refresh obs after step
        obs = self.env._state.obs

        # --- transitions ---
        def is_pos_reached(target: np.ndarray) -> np.ndarray:
            return np.linalg.norm(self.exec_pos - target, axis=1) < cfg.pos_tol

        def is_rot_reached(target_q: np.ndarray, use_real_obs: bool = False) -> np.ndarray:
            errs = np.zeros((B,), dtype=np.float32)
            for i in range(B):
                if use_real_obs:
                    rot_data = obs["ee_pose"][i, 3:]
                    if rot_data.shape[0] == 3:
                        r1 = Rotation.from_euler("xyz", rot_data, degrees=False)
                    elif rot_data.shape[0] == 4:
                        r1 = Rotation.from_quat(rot_data)
                    else:
                        r1 = Rotation.identity()
                else:
                    r1 = Rotation.from_quat(self.exec_quat[i])

                r2 = Rotation.from_quat(target_q[i])
                dq = r1 * r2.inv()
                errs[i] = np.linalg.norm(dq.as_rotvec())
                print("errs",errs)
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
            r_ok = is_rot_reached(rot_tgt, use_real_obs=check_real_rot)

            if use_rot and check_real_rot:
                reached = r_ok
                print("r_ok",r_ok)
            else:
                reached = p_ok & r_ok

            just_reached = in_state & reached & (self.state_reach_step == -1)
            if np.any(just_reached):
                self.state_reach_step[just_reached] = self.ctrl_step[just_reached]

            has_reached = self.state_reach_step != -1
            dwell_steps = int(cfg.rot_dwell_steps if use_rot else cfg.waypoint_dwell_steps)
            dwell_pass = (self.ctrl_step - self.state_reach_step) >= dwell_steps
            return in_state & has_reached & dwell_pass & reached

        # pick1
        self._enter_state(_check_and_dwell(self.ST_APP1_LIFT_Z, p_safe, q_hold), self.ST_APP1_ALIGN_X)
        self._enter_state(_check_and_dwell(self.ST_APP1_ALIGN_X, p1_x, q_hold), self.ST_APP1_ALIGN_Y)

        to_rot1 = _check_and_dwell(self.ST_APP1_ALIGN_Y, p1_y, q_hold)
        if np.any(to_rot1):
            self.rot_lock_pos[to_rot1] = self.exec_pos[to_rot1].copy()
            self._enter_state(to_rot1, self.ST_ROT_PICK1)

        done_rot1 = _check_and_dwell(self.ST_ROT_PICK1, self.rot_lock_pos, q_pick1, use_rot=True, check_real_rot=True)
        if np.any(done_rot1):
            self.hold_quat[done_rot1] = q_pick1[done_rot1].copy()
            self._enter_state(done_rot1, self.ST_PRE1_Z)

        self._enter_state(_check_and_dwell(self.ST_PRE1_Z, p1_pre, q_pick1), self.ST_PRE1_ALIGN_Y)
        self._enter_state(_check_and_dwell(self.ST_PRE1_ALIGN_Y, p1_fine_y, q_pick1), self.ST_PRE1_ALIGN_X)
        self._enter_state(_check_and_dwell(self.ST_PRE1_ALIGN_X, p1_fine_x, q_pick1), self.ST_DESCEND1)
        self._enter_state(_check_and_dwell(self.ST_DESCEND1, p1_grasp, q_pick1), self.ST_CLOSE1)

        mask_close1 = running & (s == self.ST_CLOSE1)
        if np.any(mask_close1):
            t_in = self.ctrl_step - self.state_enter_step
            done_close = t_in >= cfg.close_hold_steps
            self._enter_state(mask_close1 & done_close, self.ST_LIFT1)

        self._enter_state(_check_and_dwell(self.ST_LIFT1, p1_lift, q_pick1), self.ST_TRP1_ALIGN_X)

        # place1
        self._enter_state(_check_and_dwell(self.ST_TRP1_ALIGN_X, p_dst_hover, q_hold), self.ST_TRP1_ALIGN_Y)
        self._enter_state(_check_and_dwell(self.ST_TRP1_ALIGN_Y, p_dst_hover, q_hold), self.ST_PLACE1_DESCEND)
        self._enter_state(_check_and_dwell(self.ST_PLACE1_DESCEND, p_dst_insert, q_hold), self.ST_OPEN1)

        mask_open1 = running & (s == self.ST_OPEN1)
        if np.any(mask_open1):
            t_in = self.ctrl_step - self.state_enter_step
            done_open = t_in >= cfg.close_hold_steps
            self._enter_state(mask_open1 & done_open, self.ST_RETREAT1)

        self._enter_state(_check_and_dwell(self.ST_RETREAT1, p_retreat1, q_hold), self.ST_APP2_LIFT_Z)

        # pick2
        self._enter_state(_check_and_dwell(self.ST_APP2_LIFT_Z, p_safe, q_hold), self.ST_APP2_ALIGN_X)
        self._enter_state(_check_and_dwell(self.ST_APP2_ALIGN_X, p2_x, q_hold), self.ST_APP2_ALIGN_Y)

        to_rot2 = _check_and_dwell(self.ST_APP2_ALIGN_Y, p2_y, q_hold)
        if np.any(to_rot2):
            self.rot_lock_pos[to_rot2] = self.exec_pos[to_rot2].copy()
            self._enter_state(to_rot2, self.ST_ROT_PICK2)

        done_rot2 = _check_and_dwell(self.ST_ROT_PICK2, self.rot_lock_pos, q_pick2, use_rot=True, check_real_rot=True)
        if np.any(done_rot2):
            self.hold_quat[done_rot2] = q_pick2[done_rot2].copy()
            self._enter_state(done_rot2, self.ST_PRE2_Z)

        self._enter_state(_check_and_dwell(self.ST_PRE2_Z, p2_pre, q_pick2), self.ST_PRE2_ALIGN_Y)
        self._enter_state(_check_and_dwell(self.ST_PRE2_ALIGN_Y, p2_fine_y, q_pick2), self.ST_PRE2_ALIGN_X)
        self._enter_state(_check_and_dwell(self.ST_PRE2_ALIGN_X, p2_fine_x, q_pick2), self.ST_DESCEND2)
        self._enter_state(_check_and_dwell(self.ST_DESCEND2, p2_grasp, q_pick2), self.ST_CLOSE2)

        mask_close2 = running & (s == self.ST_CLOSE2)
        if np.any(mask_close2):
            t_in = self.ctrl_step - self.state_enter_step
            done_close = t_in >= cfg.close_hold_steps
            self._enter_state(mask_close2 & done_close, self.ST_LIFT2)

        self._enter_state(_check_and_dwell(self.ST_LIFT2, p2_lift, q_pick2), self.ST_TRP2_ALIGN_X)

        # place2
        self._enter_state(_check_and_dwell(self.ST_TRP2_ALIGN_X, p_src_hover, q_hold), self.ST_TRP2_ALIGN_Y)
        self._enter_state(_check_and_dwell(self.ST_TRP2_ALIGN_Y, p_src_hover, q_hold), self.ST_PLACE2_DESCEND)
        self._enter_state(_check_and_dwell(self.ST_PLACE2_DESCEND, p_src_insert, q_hold), self.ST_OPEN2)

        mask_open2 = running & (s == self.ST_OPEN2)
        if np.any(mask_open2):
            t_in = self.ctrl_step - self.state_enter_step
            done_open = t_in >= cfg.close_hold_steps
            self._enter_state(mask_open2 & done_open, self.ST_RETREAT2)

        self._enter_state(_check_and_dwell(self.ST_RETREAT2, p_retreat2, q_hold), self.ST_DONE)

    def collect(self) -> None:
        cfg = self.cfg
        target_n = int(cfg.data_size)
        all_ids = np.arange(self.B, dtype=np.int64)
        self.start_episodes(all_ids, seed=int(cfg.seed))

        print(f"Starting ArrangeFlowers Collection (vase swap). Target: {target_n}")

        while self.saved_count < target_n:
            self._step_logic()
            running = self.active & (~self.done)

            sample_mask = running & ((self.ctrl_step % int(cfg.sample_every_steps)) == 0)
            for env_id in np.where(sample_mask)[0]:
                self._capture_step(int(env_id))

            if cfg.save_video:
                render_mask = running & ((self.ctrl_step % int(cfg.render_every_steps)) == 0)
                for env_id in np.where(render_mask)[0]:
                    self._write_video_frame(int(env_id))

            self.ctrl_step[running] += 1

            for i in range(self.B):
                if not running[i]:
                    continue
                fsm_done = (self.states[i] == self.ST_DONE)
                timeout = (self.ctrl_step[i] >= int(cfg.max_ctrl_steps))
                env_success = bool(getattr(self.env, "success_latched", np.zeros((self.B,), dtype=bool))[i])
                if fsm_done or timeout or env_success:
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
                print(f"[Collect] Saved: {self.saved_count}/{target_n} | Active: {int(self.active.sum())}")
                self._last_log_t = now

        print(f"Done. Saved to {cfg.save_dir}")
        self.close()

    def close(self):
        # close motrix recorder (env side)
        if hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:
                pass

        # close 3DGS writers
        for vw in self.video_writers:
            if vw:
                vw.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--data_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no_video", action="store_true")

    p.add_argument("--pick1_yaw_deg", type=float, default=None)
    p.add_argument("--pick2_yaw_deg", type=float, default=None)
    p.add_argument("--pregrasp_z_margin", type=float, default=None)

    # MotrixSim toggles
    p.add_argument("--no_motrix_video", action="store_true", help="Disable MotrixSim renderer video recording.")
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
        pregrasp_z_margin=(
            args.pregrasp_z_margin if args.pregrasp_z_margin is not None else CollectorCfg.pregrasp_z_margin
        ),
        enable_motrix_video=(not args.no_motrix_video),
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
