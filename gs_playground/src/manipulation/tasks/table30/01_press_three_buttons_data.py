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

from gs_playground.src.manipulation.tasks.table30._01_press_three_buttons import (
    PressThreeButtonsEnv,
    PressThreeButtonsEnvCfg,
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def smooth_step_pos(curr: np.ndarray, tgt: np.ndarray, max_dp: float) -> np.ndarray:
    """
    curr/tgt: (B,3)
    """
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
    data_size: int = 4               # number of SUCCESS episodes to save
    num_envs: int = 2
    seed: int = 0
    save_dir: str = "./data/table30_press_three_buttons_env_collect"

    # env control
    action_mode: str = "eef"           # must match your env/robot implementation
    max_ctrl_steps: int = 200

    # motion
    max_dp: float = 0.02
    pos_tol: float = 0.02

    # press logic
    touch_thresh: float = 1e-3
    press_hold_k: int = 3

    # offsets in world frame (relative to current button body position)
    above_offset: Tuple[float, float, float] = (0.0, 0.0, 0.18)
    press_offset: Tuple[float, float, float] = (0.0, 0.0, 0.05)
    retreat_offset: Tuple[float, float, float] = (0.0, 0.0, 0.22)

    # gripper
    gripper_open: float = 0.82

    # sampling / render
    sample_every_steps: int = 1
    save_video: bool = True
    render_every_steps: int = 1
    video_fps: int = 30
    video_w: int = 800
    video_h: int = 600
    cam_view_key: str = "pixels/view_0"  # key in env obs

    # jsonl text (will be overridden per-episode with randomized order)
    subtask: str = "Press the three buttons."
    prompt: str = "press three buttons"

    # [NEW] reset randomization range (XY)
    reset_xy_range: float = 0.03


# -----------------------------------------------------------------------------
# Collector (success-only JSONL + MP4)
# -----------------------------------------------------------------------------
class PressThreeButtonsCollector:
    ST_MOVE_ABOVE = 0
    ST_DESCEND = 1
    ST_HOLD_PRESS = 2
    ST_RETRACT = 3
    ST_DONE = 4

    def __init__(self, cfg: CollectorCfg):
        self.cfg = cfg
        os.makedirs(cfg.save_dir, exist_ok=True)
        self.videos_dir = os.path.join(cfg.save_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

        env_cfg = PressThreeButtonsEnvCfg(action_mode=str(cfg.action_mode))
        self.env = PressThreeButtonsEnv(env_cfg, num_envs=int(cfg.num_envs))
        self.env.reset()

        self.model = self.env.model  # motrixsim model handle

        B = int(cfg.num_envs)
        self.B = B

        # --- per-env lifecycle ---
        self.active = np.zeros(B, dtype=bool)
        self.done = np.zeros(B, dtype=bool)
        self.success = np.zeros(B, dtype=bool)
        self.ctrl_step = np.zeros(B, dtype=np.int32)
        self._attempt_id = np.zeros(B, dtype=np.int64)

        # --- FSM + targets ---
        self.states = np.zeros(B, dtype=np.int32)

        # [CHANGED] randomized order per episode
        self.orders = np.zeros((B, 3), dtype=np.int32)   # permutation of [0,1,2]
        self.order_ptr = np.zeros(B, dtype=np.int32)     # 0..3
        self.press_hold = np.zeros(B, dtype=np.int32)
        self.pressed = np.zeros((B, 3), dtype=bool)      # pressed per button index 0..2

        # per-episode prompt/subtask
        self.ep_prompt = np.array([cfg.prompt] * B, dtype=object)
        self.ep_subtask = np.array([cfg.subtask] * B, dtype=object)

        # fixed rpy per episode (avoid orientation surprises)
        self.fixed_rpy = np.zeros((B, 3), dtype=np.float32)

        # exec target: (x,y,z,r,p,y,grip) for EEF mode
        self.exec_pos = np.zeros((B, 3), dtype=np.float32)

        # buffers for jsonl
        self.buffers: List[Dict[str, Any]] = [self._new_buffer() for _ in range(B)]

        # per-env video writer
        self.video_writers: List[Optional[EpisodeVideoWriter]] = [None] * B
        self._tmp_video_paths: List[str] = [os.path.join(self.videos_dir, f"_tmp_env{i}.mp4") for i in range(B)]

        # stats
        self.saved_success = 0
        self.attempted = 0
        self._last_log_t = time.perf_counter()

        # handles
        self.button_bodies = self.env.button_bodies
        self.button_names = tuple(self.env._cfg.button_names)            # ("button_blue", "button_green", "button_pink")
        self.button_touch_names = tuple(self.env._cfg.button_touch_names) # ("button_blue_touch", ...)

        # cached last action for ctrl logging
        self._last_action = np.zeros((B, 7), dtype=np.float32)

    @staticmethod
    def _new_buffer() -> Dict[str, Any]:
        return {
            "times": [],
            "logic_states": [],
            "qpos": [],
            "ee_pose": [],
            "gripper": [],
            "ctrl": [],
            "video_frames": 0,
        }

    def _next_seed(self, env_id: int) -> int:
        s = int(self.cfg.seed + 100000 * env_id + int(self._attempt_id[env_id]))
        self._attempt_id[env_id] += 1
        self.attempted += 1
        return s

    # ----------------------------
    # Episode start/reset
    # ----------------------------
    def start_episode(self, env_id: int, seed_i: int) -> None:
        rng = np.random.RandomState(int(seed_i))

        # reset only this env
        done_mask = np.zeros((self.B,), dtype=bool)
        done_mask[env_id] = True
        self.env.reset(done=done_mask)

        # [NEW-1] reset randomization (Collector-only)
        # Default: always randomize button bodies. Then try to randomize more bodies if available.
        # We intentionally avoid robot/table/world-like bodies to reduce risk of destabilizing the scene.
        data = self.env._state.data
        dx = float(self.cfg.reset_xy_range)

        def jitter_body_xy(body_obj) -> None:
            if not hasattr(body_obj, "get_pose") or not hasattr(body_obj, "set_pose"):
                print("not support")
                return
            try:
                pose = np.asarray(body_obj.get_pose(data[env_id]), dtype=np.float32).reshape(-1)
                if pose.shape[0] != 7:
                    return
                pose2 = pose.copy()
                pose2[0] += float(rng.uniform(-dx, dx))
                pose2[1] += float(rng.uniform(-dx, dx))
                body_obj.set_pose(data[env_id], pose2)
            except Exception:
                return

        # 1) always jitter buttons
        for b in self.button_bodies:
            jitter_body_xy(b)

        # 2) best-effort: enumerate other bodies if the model exposes names
        #    (skip if unavailable; keeps script robust across motrixsim versions)
        try:
            body_names = getattr(self.model, "body_names", None)
            if callable(body_names):
                body_names = body_names()
            if body_names is not None:
                for name in list(body_names):
                    n = str(name)
                    # skip robot/static bodies heuristically
                    if n.startswith("link") or "robotiq" in n or "panda" in n or "franka" in n:
                        continue
                    if n in ("world", "floor", "table") or "table" in n:
                        continue
                    if n in self.button_names:
                        continue
                    try:
                        body = self.model.get_body(self.model.get_body_index(n))
                        jitter_body_xy(body)
                    except Exception:
                        continue
        except Exception:
            pass

        # If FK helper exists, update kinematics after pose changes (optional)
        try:
            from motrixsim import forward_kinematic
            forward_kinematic(self.model, data[env_id:env_id + 1])
        except Exception:
            pass

        # [NEW-2] sample random order per episode and put into prompt/subtask
        ord3 = rng.permutation(3).astype(np.int32)
        self.orders[env_id] = ord3
        self.order_ptr[env_id] = 0
        self.press_hold[env_id] = 0
        self.pressed[env_id, :] = False

        n0, n1, n2 = (self.button_names[int(ord3[0])], self.button_names[int(ord3[1])], self.button_names[int(ord3[2])])
        # Keep it simple; you can localize wording if you want
        self.ep_prompt[env_id] = f"press {n0} then {n1} then {n2}"
        self.ep_subtask[env_id] = f"Press buttons in order: {n0} -> {n1} -> {n2}"

        self.active[env_id] = True
        self.done[env_id] = False
        self.success[env_id] = False
        self.ctrl_step[env_id] = 0

        self.states[env_id] = self.ST_MOVE_ABOVE

        # initialize from current obs
        obs = self.env._state.obs
        ee6 = np.asarray(obs["ee_pose"][env_id], dtype=np.float32).reshape(-1)
        if ee6.shape[0] >= 6:
            self.exec_pos[env_id] = ee6[:3]
            self.fixed_rpy[env_id] = ee6[3:6]
        else:
            self.exec_pos[env_id] = 0.0
            self.fixed_rpy[env_id] = 0.0

        # reset buffer + video tmp
        self.buffers[env_id] = self._new_buffer()

        if self.video_writers[env_id] is not None:
            self.video_writers[env_id].close()
            self.video_writers[env_id] = None

        if self.cfg.save_video:
            tmp_path = self._tmp_video_paths[env_id]
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            self.video_writers[env_id] = EpisodeVideoWriter(
                tmp_path, int(self.cfg.video_fps), (int(self.cfg.video_w), int(self.cfg.video_h))
            )

    # ----------------------------
    # Capture / write
    # ----------------------------
    def _capture_step(self, env_id: int) -> None:
        obs = self.env._state.obs

        qpos = obs["qpos"][env_id].tolist()
        ee = obs["ee_pose"][env_id].tolist()
        grip = obs["gripper"][env_id].tolist()

        ctrl_vec = self._last_action[env_id].tolist()
        t_sec = float(self.ctrl_step[env_id] * float(self.env._cfg.ctrl_dt))

        self.buffers[env_id]["times"].append(t_sec)
        self.buffers[env_id]["logic_states"].append(int(self.states[env_id]))
        self.buffers[env_id]["qpos"].append(qpos)
        self.buffers[env_id]["ee_pose"].append(ee)
        self.buffers[env_id]["gripper"].append(grip)
        self.buffers[env_id]["ctrl"].append(ctrl_vec)

    def _write_video_frame(self, env_id: int) -> None:
        vw = self.video_writers[env_id]
        if vw is None:
            return
        obs = self.env._state.obs
        if self.cfg.cam_view_key not in obs:
            return
        rgb = obs[self.cfg.cam_view_key][env_id]
        if rgb is None:
            return
        bgr = rgb[..., ::-1].copy()
        vw.write(bgr)
        self.buffers[env_id]["video_frames"] += 1

    def _flush_episode_jsonl(self, env_id: int, ep_idx: int, video_rel_path: str) -> None:
        path = os.path.join(self.cfg.save_dir, f"episode_{ep_idx:05d}.jsonl")
        buf = self.buffers[env_id]
        n = len(buf["times"])
        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                legacy_state = buf["qpos"][i] + buf["gripper"][i]
                rec = {
                    "images_1": {"url": video_rel_path, "type": "video", "frame_idx": i},
                    "subtask": str(self.ep_subtask[env_id]),
                    "prompt": str(self.ep_prompt[env_id]),
                    "state": legacy_state,
                    "qpos": buf["qpos"][i],
                    "ee_pose": buf["ee_pose"][i],
                    "ctrl": buf["ctrl"][i],
                    "gripper": buf["gripper"][i],
                    "is_robot": True,
                    "logic_state": int(buf["logic_states"][i]),
                    "time": float(buf["times"][i]),
                    "success": bool(self.success[env_id]),
                    "order": self.orders[env_id].astype(int).tolist(),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _finalize_episode(self, env_id: int) -> None:
        if self.video_writers[env_id] is not None:
            self.video_writers[env_id].close()
            self.video_writers[env_id] = None

        tmp_path = self._tmp_video_paths[env_id]

        if self.success[env_id] and (self.saved_success < int(self.cfg.data_size)):
            ep_idx = int(self.saved_success)
            final_video_abs = os.path.join(self.videos_dir, f"episode_{ep_idx:05d}.mp4")
            video_rel_path = f"videos/episode_{ep_idx:05d}.mp4"

            if self.cfg.save_video:
                try:
                    if os.path.exists(final_video_abs):
                        os.remove(final_video_abs)
                except Exception:
                    pass
                try:
                    os.replace(tmp_path, final_video_abs)
                except Exception:
                    try:
                        shutil.copy2(tmp_path, final_video_abs)
                        os.remove(tmp_path)
                    except Exception:
                        pass

            self._flush_episode_jsonl(env_id, ep_idx, video_rel_path)
            self.saved_success += 1
        else:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

        self.buffers[env_id] = self._new_buffer()

    # ----------------------------
    # Core logic (vectorized)
    # ----------------------------
    def _step_logic(self) -> None:
        cfg = self.cfg
        B = self.B

        running = self.active & (~self.done)
        if not np.any(running):
            return

        data = self.env._state.data

        # button poses (B,3,7) -> pos (B,3,3)
        btn_poses_list = [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.button_bodies]
        btn_poses = np.stack(btn_poses_list, axis=1)
        btn_pos = btn_poses[..., :3]

        # [CHANGED] current target button is from randomized order
        ptr = np.clip(self.order_ptr, 0, 2)
        cur_btn = self.orders[np.arange(B), ptr]               # (B,)
        cur_btn = np.clip(cur_btn, 0, 2)
        cur_btn_pos = btn_pos[np.arange(B), cur_btn]           # (B,3)

        above_p = cur_btn_pos + np.asarray(cfg.above_offset, dtype=np.float32).reshape(1, 3)
        press_p = cur_btn_pos + np.asarray(cfg.press_offset, dtype=np.float32).reshape(1, 3)
        retreat_p = cur_btn_pos + np.asarray(cfg.retreat_offset, dtype=np.float32).reshape(1, 3)

        # choose target by state
        tgt_pos = self.exec_pos.copy()
        s = self.states

        m0 = running & (s == self.ST_MOVE_ABOVE)
        m1 = running & (s == self.ST_DESCEND)
        m2 = running & (s == self.ST_HOLD_PRESS)
        m3 = running & (s == self.ST_RETRACT)

        if np.any(m0):
            tgt_pos[m0] = above_p[m0]
        if np.any(m1):
            tgt_pos[m1] = press_p[m1]
        if np.any(m2):
            tgt_pos[m2] = press_p[m2]
        if np.any(m3):
            tgt_pos[m3] = retreat_p[m3]

        # smooth
        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos, float(cfg.max_dp))

        # build action: [x,y,z,r,p,y,grip]
        action = np.zeros((B, 7), dtype=np.float32)
        action[:, :3] = self.exec_pos
        action[:, 3:6] = self.fixed_rpy
        action[:, 6] = float(cfg.gripper_open)

        self._last_action[:] = action
        self.env.step(action)

        # read all touch sensors -> (B,3)
        touch_all = []
        for sn in self.button_touch_names:
            v = np.asarray(self.model.get_sensor_value(sn, data), dtype=np.float32)
            v = v.reshape(B, -1)[:, 0].astype(np.float32)
            touch_all.append(v)
        touch_all = np.stack(touch_all, axis=1)

        # pick touch value for current target
        touch = touch_all[np.arange(B), cur_btn].astype(np.float32)

        # reached checks (use exec_pos)
        reach_above = np.linalg.norm(self.exec_pos - above_p, axis=1) < float(cfg.pos_tol)
        reach_press = np.linalg.norm(self.exec_pos - press_p, axis=1) < float(cfg.pos_tol)
        reach_retreat = np.linalg.norm(self.exec_pos - retreat_p, axis=1) < float(cfg.pos_tol)

        # transitions
        to_descend = running & (s == self.ST_MOVE_ABOVE) & reach_above
        if np.any(to_descend):
            self.states[to_descend] = self.ST_DESCEND
            self.press_hold[to_descend] = 0

        to_hold = running & (s == self.ST_DESCEND) & reach_press
        if np.any(to_hold):
            self.states[to_hold] = self.ST_HOLD_PRESS
            self.press_hold[to_hold] = 0

        holding = running & (self.states == self.ST_HOLD_PRESS)
        if np.any(holding):
            ok_touch = touch > float(cfg.touch_thresh)
            ok = holding & ok_touch & reach_press
            self.press_hold[holding] = np.where(ok[holding], self.press_hold[holding] + 1, 0)

        pressed_now = running & (self.states == self.ST_HOLD_PRESS) & (self.press_hold >= int(cfg.press_hold_k))
        if np.any(pressed_now):
            # mark the CURRENT BUTTON (cur_btn), not stage index
            self.pressed[np.arange(B), cur_btn] |= pressed_now
            self.press_hold[pressed_now] = 0
            self.states[pressed_now] = self.ST_RETRACT

        to_next = running & (self.states == self.ST_RETRACT) & reach_retreat
        if np.any(to_next):
            self.order_ptr[to_next] += 1
            finished = to_next & (self.order_ptr >= 3)
            if np.any(finished):
                self.states[finished] = self.ST_DONE
            cont = to_next & (~finished)
            if np.any(cont):
                self.states[cont] = self.ST_MOVE_ABOVE

    # ----------------------------
    # Collect loop
    # ----------------------------
    def collect(self) -> None:
        cfg = self.cfg
        target = int(cfg.data_size)

        # start all envs
        for i in range(self.B):
            self.start_episode(i, self._next_seed(i))

        while self.saved_success < target:
            self._step_logic()

            running = self.active & (~self.done)

            # sample
            sample_mask = running & ((self.ctrl_step % int(cfg.sample_every_steps)) == 0)
            for env_id in np.where(sample_mask)[0].tolist():
                self._capture_step(env_id)

            # render
            if cfg.save_video:
                render_mask = running & ((self.ctrl_step % int(cfg.render_every_steps)) == 0)
                for env_id in np.where(render_mask)[0].tolist():
                    self._write_video_frame(env_id)

            # step counter
            self.ctrl_step[running] += 1

            # terminal checks
            for i in range(self.B):
                if (not self.active[i]) or self.done[i]:
                    continue
                timeout = int(self.ctrl_step[i]) >= int(cfg.max_ctrl_steps)
                finished = int(self.states[i]) == self.ST_DONE
                if finished or timeout:
                    self.done[i] = True
                    self.success[i] = bool(finished and np.all(self.pressed[i]))

            # finalize & restart
            for i in range(self.B):
                if (not self.active[i]) or (not self.done[i]):
                    continue

                self._finalize_episode(i)

                if self.saved_success >= target:
                    self.active[i] = False
                    continue

                self.start_episode(i, self._next_seed(i))

            # log
            now = time.perf_counter()
            if (now - self._last_log_t) >= 2.0:
                print(
                    f"[collect] active={int(np.sum(self.active))}/{self.B} "
                    f"saved_success={int(self.saved_success)}/{target} attempted={int(self.attempted)}"
                )
                self._last_log_t = now

        print(f"[DONE] saved_success={self.saved_success}/{target}, attempted={self.attempted}")
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
    p.add_argument("--action_mode", type=str, default=None, choices=["eef", "joint"])
    p.add_argument("--max_ctrl_steps", type=int, default=None)
    p.add_argument("--reset_xy_range", type=float, default=None)
    args = p.parse_args()

    cfg = CollectorCfg(
        save_dir=args.save_dir if args.save_dir is not None else CollectorCfg.save_dir,
        data_size=args.data_size if args.data_size is not None else CollectorCfg.data_size,
        num_envs=args.num_envs if args.num_envs is not None else CollectorCfg.num_envs,
        seed=args.seed if args.seed is not None else CollectorCfg.seed,
        save_video=(not args.no_video),
        action_mode=args.action_mode if args.action_mode is not None else CollectorCfg.action_mode,
        max_ctrl_steps=args.max_ctrl_steps if args.max_ctrl_steps is not None else CollectorCfg.max_ctrl_steps,
        reset_xy_range=args.reset_xy_range if args.reset_xy_range is not None else CollectorCfg.reset_xy_range,
    )

    runner = PressThreeButtonsCollector(cfg)
    try:
        runner.collect()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
