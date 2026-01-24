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

from gs_playground.src.manipulation.tasks.table30._04_hang_toothbrush_cup import (
    HangToothbrushCupEnv,
    HangToothbrushCupEnvCfg,
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def smooth_step_pos(curr: np.ndarray, tgt: np.ndarray, max_dp: float) -> np.ndarray:
    """Smooth position update with max step."""
    dp = tgt - curr
    n = np.linalg.norm(dp, axis=1, keepdims=True)
    s = np.minimum(1.0, float(max_dp) / (n + 1e-9))
    return curr + dp * s


def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    sym = np.pi
    return (a + sym / 2.0) % sym - sym / 2.0


def closest_yaw(target: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """
    Map target yaw to the equivalent angle (2π-periodic) that is closest to curr yaw.
    """
    d = wrap_to_pi(target - curr)
    return curr + d



def to_xyzw_from_possible_wxyz(q: np.ndarray, assume: str = "xyzw") -> np.ndarray:
    """
    q: (...,4)
    assume:
      - "xyzw": already scipy format
      - "wxyz": convert [w,x,y,z] -> [x,y,z,w]
    """
    if assume == "xyzw":
        return q
    if assume == "wxyz":
        return q[..., [1, 2, 3, 0]]
    raise ValueError(f"Unknown assume={assume}")


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
    data_size: int = 10
    num_envs: int = 15

    seed: int = 1500
    save_dir: str = "./data/table30_hang_toothbrush_cup_collect_yaw_stack_style_debug"

    # env control
    max_ctrl_steps: int = 1200

    # motion position
    max_dp: float = 0.005
    pos_tol: float = 0.001

    # rotation control
    rot_gain: float = 0.6
    max_dr: float = 0.08
    yaw_tol: float = 0.02

    # yaw offsets
    yaw_offset_grasp: float = 0.0

    # keypoints offsets (World Frame)
    grasp_offset: Tuple[float, float, float] = (0.0, 0.0, 0.02)
    pre_grasp_z: float = 0.05

    pre_hang_offset: Tuple[float, float, float] = (-0.046, -0.15, 0.03)
    hang_offset: Tuple[float, float, float] = (-0.046, -0.02, 0.03)
    retreat_dx: float = 0.10

    # gripper
    gripper_open: float = 0.0
    gripper_close: float = 0.8

    # timing / dwell
    close_hold_steps: int = 25
    release_hold_steps: int = 25
    waypoint_dwell_steps: int = 20

    # sampling / render
    sample_every_steps: int = 1
    save_video: bool = True
    render_every_steps: int = 1
    video_fps: int = 30
    video_w: int = 640
    video_h: int = 480
    cam_view_key: Optional[str] = None

    # quat convention for sites/poses (IMPORTANT)
    # If your get_pose returns [qx,qy,qz,qw], use "xyzw"
    # If it returns [qw,qx,qy,qz], use "wxyz"
    site_quat_convention: str = "xyzw"  # change to "wxyz" if needed

    # text fields
    subtask: Optional[str] = None
    prompt: Optional[str] = None


# -----------------------------------------------------------------------------
# Collector
# -----------------------------------------------------------------------------
class HangToothbrushCupCollector:
    """
    “Stack-style yaw logic” for toothbrush cup:
      - Before descend: rotate to cup yaw (from grasp site)
      - After close and lift to safe transport Z: rotate back to start yaw (== rotate -delta)
      - Rest Manhattan position logic matches the no-rotation version.
      - No symmetry wrapping (cup is non-symmetric).
    """

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
    ST_TRP_UNYAW = 7
    ST_TRP_ALIGN_X = 8
    ST_TRP_ALIGN_Y = 9
    ST_HANG_DOWN = 10

    # Phase 4: Release & End
    ST_RELEASE = 11
    ST_RETREAT = 12
    ST_GO_RESET = 13
    ST_DONE = 14

    def __init__(self, cfg: CollectorCfg, env_cfg: Optional[HangToothbrushCupEnvCfg] = None):
        self.cfg = cfg
        os.makedirs(cfg.save_dir, exist_ok=True)
        self.videos_dir = os.path.join(cfg.save_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

        self.env_cfg = env_cfg if env_cfg is not None else HangToothbrushCupEnvCfg()
        self.env = HangToothbrushCupEnv(self.env_cfg, num_envs=int(cfg.num_envs))
        self.env.reset()

        self.model = self.env.model
        self.B = int(cfg.num_envs)
        self.cam_view_key = cfg.cam_view_key or "pixels/view_0"

        instruction = str(getattr(self.env._cfg, "instruction", "") or "")
        self.ep_subtask = np.array([cfg.subtask or instruction] * self.B, dtype=object)
        self.ep_prompt = np.array([cfg.prompt or (instruction if instruction else "hang toothbrush cup")] * self.B, dtype=object)

        B = self.B
        self.active = np.zeros(B, dtype=bool)
        self.done = np.zeros(B, dtype=bool)
        self.success = np.zeros(B, dtype=bool)
        self.ctrl_step = np.zeros(B, dtype=np.int32)
        self._attempt_id = np.zeros(B, dtype=np.int64)

        self.states = np.zeros(B, dtype=np.int32)
        self.state_enter_step = np.zeros(B, dtype=np.int32)
        self.state_reach_step = np.full(B, -1, dtype=np.int32)

        # Controls
        self.exec_pos = np.zeros((B, 3), dtype=np.float32)
        self.exec_quat = np.zeros((B, 4), dtype=np.float32)  # xyzw

        self.home_pos = np.zeros((B, 3), dtype=np.float32)

        # Latch positions
        self.latched_start_pos = np.zeros((B, 3), dtype=np.float32)
        self.latched_grasp_pos = np.zeros((B, 3), dtype=np.float32)
        self.latched_hook_pos = np.zeros((B, 3), dtype=np.float32)

        # Latch yaw info
        self.latched_start_quat = np.zeros((B, 4), dtype=np.float32)
        self.latched_start_yaw = np.zeros((B,), dtype=np.float32)
        self.latched_cup_yaw = np.zeros((B,), dtype=np.float32)

        self.buffers: List[Dict[str, Any]] = [self._new_buffer() for _ in range(B)]
        self.video_writers: List[Optional[EpisodeVideoWriter]] = [None] * B
        self._tmp_video_paths: List[str] = [os.path.join(self.videos_dir, f"_tmp_env{i}.mp4") for i in range(B)]

        self.saved_success = 0
        self.attempted = 0
        self._last_log_t = time.perf_counter()
        self._last_action = np.zeros((B, 7), dtype=np.float32)

        self.grasp_site = self.env.grasp_site
        self.hook_site = self.env.hook_site

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

        self.active[env_ids] = True
        self.done[env_ids] = False
        self.success[env_ids] = False
        self.ctrl_step[env_ids] = 0
        self.state_enter_step[env_ids] = 0
        self.state_reach_step[env_ids] = -1

        data = self.env._state.data
        self.env.robot.reset_envs(data, done_mask)
        self.env.robot.update_reference(data)

        # --- Robot start pose ---
        ee_pose_raw = self.env.robot.get_ee_pose(data)  # (B,6) or (B,7)
        self.exec_pos[env_ids] = ee_pose_raw[env_ids, :3]
        self.latched_start_pos[env_ids] = ee_pose_raw[env_ids, :3]
        self.home_pos[env_ids] = ee_pose_raw[env_ids, :3]

        if ee_pose_raw.shape[1] == 6:
            r = Rotation.from_euler("xyz", ee_pose_raw[env_ids, 3:6], degrees=False)
            self.exec_quat[env_ids] = r.as_quat().astype(np.float32)
        else:
            self.exec_quat[env_ids] = ee_pose_raw[env_ids, 3:7].astype(np.float32)

        self.latched_start_quat[env_ids] = self.exec_quat[env_ids].copy()
        self.latched_start_yaw[env_ids] = Rotation.from_quat(self.exec_quat[env_ids]).as_euler("xyz", degrees=False)[:, 2].astype(
            np.float32
        )

        # --- Site poses (grasp/hook) ---
        grasp_pose7 = np.asarray(self.grasp_site.get_pose(data), dtype=np.float32).reshape(self.B, -1)
        hook_pose7 = np.asarray(self.hook_site.get_pose(data), dtype=np.float32).reshape(self.B, -1)

        self.latched_grasp_pos[env_ids] = grasp_pose7[env_ids, :3]
        self.latched_hook_pos[env_ids] = hook_pose7[env_ids, :3]

        # Cup yaw from grasp site quaternion
        grasp_q = grasp_pose7[env_ids, 3:7]
        grasp_q = to_xyzw_from_possible_wxyz(grasp_q, assume=str(self.cfg.site_quat_convention))
        cup_yaw = Rotation.from_quat(grasp_q).as_euler("xyz", degrees=False)[:, 2]
        self.latched_cup_yaw[env_ids] = cup_yaw.astype(np.float32)

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
        info = self.env._state.info

        buf = self.buffers[env_id]
        buf["times"].append(float(self.ctrl_step[env_id] * 0.02))
        buf["logic_states"].append(int(self.states[env_id]))
        buf["qpos"].append(obs["qpos"][env_id].tolist())
        buf["ee_pose"].append(obs["ee_pose"][env_id].tolist())
        buf["gripper"].append(obs["gripper"][env_id].tolist())
        buf["ctrl"].append(self._last_action[env_id].tolist())
        buf["reward"].append(float(np.asarray(self.env._state.reward).reshape(-1)[env_id]))

        is_success = bool(np.asarray(info.get("is_success", False)).reshape(-1)[env_id])
        buf["is_success"].append(is_success)

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
        prompt = str(self.ep_prompt[env_id])

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
                    # include per-episode latched yaw in first frame if helpful
                    # (you can remove if not needed)
                    **(
                        {
                            "latched_cup_yaw": float(self.latched_cup_yaw[env_id]),
                            "latched_start_yaw": float(self.latched_start_yaw[env_id]),
                        }
                        if i == 0
                        else {}
                    ),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _finalize_episode(self, env_id: int) -> None:
        if self.video_writers[env_id]:
            self.video_writers[env_id].close()
            self.video_writers[env_id] = None

        if self.success[env_id] and (self.saved_success < int(self.cfg.data_size)):
        # if True :
            ep_idx = int(self.saved_success)
            final_video_path = f"videos/episode_{ep_idx:05d}.mp4"
            abs_video_path = os.path.join(self.cfg.save_dir, final_video_path)

            if self.cfg.save_video and os.path.exists(self._tmp_video_paths[env_id]):
                shutil.move(self._tmp_video_paths[env_id], abs_video_path)

            self._flush_jsonl(env_id, ep_idx, final_video_path)
            self.saved_success += 1
            print(f"[Saved] episode {ep_idx}. Total saved: {self.saved_success}")

        if os.path.exists(self._tmp_video_paths[env_id]):
            try:
                os.remove(self._tmp_video_paths[env_id])
            except Exception:
                pass
        self.buffers[env_id] = self._new_buffer()

    # ----------------------------
    # Core Logic
    # ----------------------------
    def _enter_state(self, mask: np.ndarray, new_state: int) -> None:
        if not np.any(mask):
            return
        self.states[mask] = int(new_state)
        self.state_enter_step[mask] = self.ctrl_step[mask].copy()
        self.state_reach_step[mask] = -1

    def _step_logic(self) -> None:
        cfg = self.cfg
        B = self.B

        running = self.active & (~self.done)
        if not np.any(running):
            return

        # -------------------------
        # 1) Manhattan keypoints (same as no-rotation version)
        # -------------------------
        start_p = self.latched_start_pos
        grasp_p = self.latched_grasp_pos + np.asarray(cfg.grasp_offset, dtype=np.float32).reshape(1, 3)
        hook_p = self.latched_hook_pos

        safe_z = grasp_p[:, 2] + float(cfg.pre_grasp_z)

        p_app_lift_z = start_p.copy()
        p_app_lift_z[:, 2] = safe_z

        p_app_align_x = p_app_lift_z.copy()
        p_app_align_x[:, 0] = grasp_p[:, 0]

        p_app_align_y = grasp_p.copy()
        p_app_align_y[:, 2] = safe_z

        p_app_descend = grasp_p

        pre_hang_p = hook_p + np.asarray(cfg.pre_hang_offset, dtype=np.float32).reshape(1, 3)
        hang_p = hook_p + np.asarray(cfg.hang_offset, dtype=np.float32).reshape(1, 3)

        p_trp_lift_z = grasp_p.copy()
        p_trp_lift_z[:, 2] = pre_hang_p[:, 2]

        p_trp_align_x = p_trp_lift_z.copy()
        p_trp_align_x[:, 0] = pre_hang_p[:, 0]

        p_trp_align_y = pre_hang_p

        retreat_p = hang_p.copy()
        retreat_p[:, 0] -= float(cfg.retreat_dx)

        reset_p = self.home_pos

        # -------------------------
        # 2) Target assignment (pos + yaw only in yaw states)
        # -------------------------
        s = self.states

        tgt_pos = self.exec_pos.copy()
        tgt_yaw = np.full((B,), np.nan, dtype=np.float32)
        grip_cmd = np.full((B,), float(cfg.gripper_open), dtype=np.float32)

        def set_target(state_id, pos, grip):
            mask = running & (s == state_id)
            if np.any(mask):
                tgt_pos[mask] = pos[mask]
                grip_cmd[mask] = float(grip)

        def set_yaw_target(state_id, yaw_arr):
            mask = running & (s == state_id)
            if np.any(mask):
                tgt_yaw[mask] = yaw_arr[mask]

        # Approach
        set_target(self.ST_APP_LIFT_Z, p_app_lift_z, cfg.gripper_open)
        set_target(self.ST_APP_ALIGN_X, p_app_align_x, cfg.gripper_open)
        set_target(self.ST_APP_ALIGN_Y, p_app_align_y, cfg.gripper_open)

        # Yaw align to cup
        set_target(self.ST_APP_ALIGN_YAW, p_app_align_y, cfg.gripper_open)
        yaw_grasp = (self.latched_cup_yaw + float(cfg.yaw_offset_grasp)).astype(np.float32) * -1
        set_yaw_target(self.ST_APP_ALIGN_YAW, yaw_grasp)

        set_target(self.ST_APP_DESCEND, p_app_descend, cfg.gripper_open)

        # Close
        set_target(self.ST_CLOSE, p_app_descend, cfg.gripper_close)

        # Transport
        set_target(self.ST_TRP_LIFT_Z, p_trp_lift_z, cfg.gripper_close)

        # Unyaw back to start yaw
        set_target(self.ST_TRP_UNYAW, p_trp_lift_z, cfg.gripper_close)
        yaw_back = self.latched_start_yaw.astype(np.float32) * -1
        set_yaw_target(self.ST_TRP_UNYAW, yaw_back)

        set_target(self.ST_TRP_ALIGN_X, p_trp_align_x, cfg.gripper_close)
        set_target(self.ST_TRP_ALIGN_Y, p_trp_align_y, cfg.gripper_close)
        set_target(self.ST_HANG_DOWN, hang_p, cfg.gripper_close)

        # Release & End
        set_target(self.ST_RELEASE, hang_p, cfg.gripper_open)
        set_target(self.ST_RETREAT, retreat_p, cfg.gripper_open)
        set_target(self.ST_GO_RESET, reset_p, cfg.gripper_open)

        # -------------------------
        # 3) Compute control: pos + yaw rotvec (stack-style)
        # -------------------------
        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos, float(cfg.max_dp))

        # Reference pose
        ref_pose = self.env.robot.ref_ee_pose
        ref_pos = ref_pose[:, :3]

        if ref_pose.shape[1] == 6:
            ref_quat = Rotation.from_euler("xyz", ref_pose[:, 3:6], degrees=False).as_quat().astype(np.float32)
        else:
            ref_quat = ref_pose[:, 3:7].astype(np.float32)

        want_rot = running & (~np.isnan(tgt_yaw))
        desired_quat = self.exec_quat.copy()

        # ---- compute curr_yaw early (needed for closest_yaw) ----
        obs = self.env._state.obs
        ee_pose = obs.get("ee_pose", None)
        if ee_pose is not None and ee_pose.shape[1] == 7:
            ee_quat = ee_pose[:, 3:7].astype(np.float32)
        elif ee_pose is not None and ee_pose.shape[1] == 6:
            ee_quat = Rotation.from_euler("xyz", ee_pose[:, 3:6], degrees=False).as_quat().astype(np.float32)
        else:
            ee_quat = ref_quat.copy()

        curr_yaw = Rotation.from_quat(ee_quat).as_euler("xyz", degrees=False)[:, 2].astype(np.float32)

        if np.any(want_rot):
            # raw targets for the active envs
            target_y_raw = tgt_yaw[want_rot].astype(np.float32)

            # IMPORTANT: map to closest branch around current yaw (avoid +/-pi jumps)
            target_y = closest_yaw(target_y_raw, curr_yaw[want_rot]).astype(np.float32)

            start_y = self.latched_start_yaw[want_rot].astype(np.float32)
            delta_y = wrap_to_pi(target_y - start_y).astype(np.float32)

            r_delta = Rotation.from_euler("z", delta_y)
            r_start = Rotation.from_quat(self.latched_start_quat[want_rot])
            r_target = r_delta * r_start  # stack-style composition

            desired_quat[want_rot] = r_target.as_quat().astype(np.float32)
            self.exec_quat[want_rot] = desired_quat[want_rot]


        rotvec_cmd = np.zeros((B, 3), dtype=np.float32)
        if np.any(want_rot):
            r_des = Rotation.from_quat(desired_quat[want_rot])
            r_ref = Rotation.from_quat(ref_quat[want_rot])

            # Stack-style error definition:
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

        # -------------------------
        # 4) Checks & FSM transitions (pos dwell + yaw dwell)
        # -------------------------
        def is_reached(p: np.ndarray) -> np.ndarray:
            return np.linalg.norm(self.exec_pos - p, axis=1) < float(cfg.pos_tol)


        # if np.any(want_rot):
        #     i = np.where(want_rot)[0][0]
        #     print(
        #         "start_yaw:", float(self.latched_start_yaw[i]),
        #         "target_yaw:", float(tgt_yaw[i]),
        #         "delta:", float(wrap_to_pi(tgt_yaw[i] - self.latched_start_yaw[i])),
        #         "curr_yaw:", float(curr_yaw[i]),
        #     )


        def _check_reach_dwell(state_idx, p):
            in_s = running & (self.states == state_idx)
            ok = is_reached(p)

            just = in_s & ok & (self.state_reach_step == -1)
            self.state_reach_step[just] = self.ctrl_step[just]

            dwell = (self.ctrl_step - self.state_reach_step) >= int(cfg.waypoint_dwell_steps)
            return in_s & (self.state_reach_step != -1) & dwell & ok

        def _check_yaw_dwell(state_idx, y):
            in_s = running & (self.states == state_idx)
            dy = wrap_to_pi(curr_yaw - y)
            ok = np.abs(dy) < float(cfg.yaw_tol)

            just = in_s & ok & (self.state_reach_step == -1)
            self.state_reach_step[just] = self.ctrl_step[just]

            dwell = (self.ctrl_step - self.state_reach_step) >= int(cfg.waypoint_dwell_steps)
            return in_s & (self.state_reach_step != -1) & dwell & ok

        # Phase 1
        self._enter_state(_check_reach_dwell(self.ST_APP_LIFT_Z, p_app_lift_z), self.ST_APP_ALIGN_X)
        self._enter_state(_check_reach_dwell(self.ST_APP_ALIGN_X, p_app_align_x), self.ST_APP_ALIGN_Y)
        self._enter_state(_check_reach_dwell(self.ST_APP_ALIGN_Y, p_app_align_y), self.ST_APP_ALIGN_YAW)

        done_app_yaw = _check_yaw_dwell(self.ST_APP_ALIGN_YAW, yaw_grasp)
        self._enter_state(done_app_yaw, self.ST_APP_DESCEND)

        self._enter_state(_check_reach_dwell(self.ST_APP_DESCEND, p_app_descend), self.ST_CLOSE)

        # Close (timer)
        mask_close = running & (self.states == self.ST_CLOSE)
        if np.any(mask_close):
            done_close = (self.ctrl_step - self.state_enter_step) >= int(cfg.close_hold_steps)
            self._enter_state(mask_close & done_close, self.ST_TRP_LIFT_Z)

        # Transport
        done_trp_lift = _check_reach_dwell(self.ST_TRP_LIFT_Z, p_trp_lift_z)
        self._enter_state(done_trp_lift, self.ST_TRP_UNYAW)

        done_unyaw = _check_yaw_dwell(self.ST_TRP_UNYAW, yaw_back)
        self._enter_state(done_unyaw, self.ST_TRP_ALIGN_X)

        self._enter_state(_check_reach_dwell(self.ST_TRP_ALIGN_X, p_trp_align_x), self.ST_TRP_ALIGN_Y)
        self._enter_state(_check_reach_dwell(self.ST_TRP_ALIGN_Y, p_trp_align_y), self.ST_HANG_DOWN)
        self._enter_state(_check_reach_dwell(self.ST_HANG_DOWN, hang_p), self.ST_RELEASE)

        # Release & End
        in_rel = running & (self.states == self.ST_RELEASE)
        done_rel = in_rel & ((self.ctrl_step - self.state_enter_step) >= int(cfg.release_hold_steps))
        self._enter_state(done_rel, self.ST_RETREAT)

        done_ret = is_reached(retreat_p)
        self._enter_state(running & (self.states == self.ST_RETREAT) & done_ret, self.ST_GO_RESET)

        done_rst = is_reached(reset_p)
        self._enter_state(running & (self.states == self.ST_GO_RESET) & done_rst, self.ST_DONE)

        # Success check
        info = self.env._state.info
        is_success = np.asarray(
            info.get("is_success", info.get("success", np.zeros((B,), dtype=np.bool_))),
            dtype=np.bool_,
        ).reshape(-1)

        finished = running & is_success
        if np.any(finished):
            self.states[finished] = self.ST_DONE

    def collect(self) -> None:
        cfg = self.cfg
        target = int(cfg.data_size)

        all_ids = np.arange(self.B, dtype=np.int64)
        self.start_episodes(all_ids, seed=int(cfg.seed))

        # print(f"Starting Collection. Target: {target}")

        while self.saved_success < target:
            self._step_logic()

            running = self.active & (~self.done)
            sample_mask = running & ((self.ctrl_step % int(cfg.sample_every_steps)) == 0)
            for env_id in np.where(sample_mask)[0].tolist():
                self._capture_step(env_id)

            if cfg.save_video:
                render_mask = running & ((self.ctrl_step % int(cfg.render_every_steps)) == 0)
                for env_id in np.where(render_mask)[0].tolist():
                    self._write_video_frame(env_id)

            self.ctrl_step[running] += 1

            info = self.env._state.info
            is_success = np.asarray(
                info.get("is_success", info.get("success", np.zeros((self.B,), dtype=np.bool_))),
                dtype=np.bool_,
            ).reshape(-1)

            for i in range(self.B):
                if (not self.active[i]) or self.done[i]:
                    continue
                timeout = int(self.ctrl_step[i]) >= int(cfg.max_ctrl_steps)
                finished = bool(is_success[i]) or (int(self.states[i]) == int(self.ST_DONE))
                if finished or timeout:
                    self.done[i] = True
                    self.success[i] = bool(is_success[i])

            done_ids = np.where(self.active & self.done)[0]
            for env_id in done_ids.tolist():
                self._finalize_episode(int(env_id))

            if self.saved_success >= target:
                self.active[done_ids] = False
            else:
                restart_ids = done_ids[self.active[done_ids]]
                if restart_ids.size > 0:
                    batch_seed = int(cfg.seed + self.attempted)
                    self.done[restart_ids] = False
                    self.start_episodes(restart_ids, seed=batch_seed)

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
        for vw in self.video_writers:
            if vw is not None:
                vw.close()


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
    p.add_argument("--site_quat_convention", type=str, default=None, choices=["xyzw", "wxyz"])
    args = p.parse_args()

    env_cfg = HangToothbrushCupEnvCfg()
    if args.action_mode is not None:
        env_cfg.action_mode = str(args.action_mode)

    cfg = CollectorCfg(
        save_dir=args.save_dir if args.save_dir is not None else CollectorCfg.save_dir,
        data_size=args.data_size if args.data_size is not None else CollectorCfg.data_size,
        num_envs=args.num_envs if args.num_envs is not None else CollectorCfg.num_envs,
        seed=args.seed if args.seed is not None else CollectorCfg.seed,
        save_video=(not args.no_video),
        max_ctrl_steps=args.max_ctrl_steps if args.max_ctrl_steps is not None else CollectorCfg.max_ctrl_steps,
        site_quat_convention=args.site_quat_convention if args.site_quat_convention else CollectorCfg.site_quat_convention,
    )

    runner = HangToothbrushCupCollector(cfg, env_cfg=env_cfg)
    try:
        runner.collect()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
