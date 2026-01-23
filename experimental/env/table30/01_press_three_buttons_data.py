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
    data_size: int = 4
    num_envs: int = 1
    seed: int = 0
    save_dir: str = "./data/table30_press_three_buttons_env_collect"

    # env control
    action_mode: str = "eef"
    max_ctrl_steps: int = 500

    # motion
    max_dp: float = 0.01
    pos_tol: float = 0.02

    # press logic
    touch_thresh: float = 1e-3

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
    cam_view_key: str = "pixels/view_0"

    # text fields (fixed; 不再 per-episode 随机覆盖)
    subtask: str = "Press the three buttons in sequence."
    prompt: str = "press three buttons"


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

        self.model = self.env.model
        self.B = int(cfg.num_envs)

        # --- per-env lifecycle ---
        B = self.B
        self.active = np.zeros(B, dtype=bool)
        self.done = np.zeros(B, dtype=bool)
        self.success = np.zeros(B, dtype=bool)
        self.ctrl_step = np.zeros(B, dtype=np.int32)
        self._attempt_id = np.zeros(B, dtype=np.int64)

        # --- FSM + targets ---
        self.states = np.zeros(B, dtype=np.int32)

        # fixed rpy per episode
        self.fixed_rpy = np.zeros((B, 3), dtype=np.float32)

        # exec target position (x,y,z) for EEF
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
        self.button_names = tuple(self.env._cfg.button_names)
        self.button_touch_names = tuple(self.env._cfg.button_touch_names)

        # cached last action for ctrl logging
        self._last_action = np.zeros((B, 7), dtype=np.float32)

        # per-episode prompt/subtask (fixed)
        self.ep_prompt = np.array([cfg.prompt] * B, dtype=object)
        self.ep_subtask = np.array([cfg.subtask] * B, dtype=object)

    @staticmethod
    def _new_buffer() -> Dict[str, Any]:
        return {
            "times": [],
            "logic_states": [],
            "qpos": [],
            "ee_pose": [],
            "gripper": [],
            "ctrl": [],
            # --- reuse env info ---
            "reward": [],
            "cur_idx": [],
            "completed": [],
            "dist": [],
            "touch_val": [],
            "is_success": [],
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
        # reset only this env
        done_mask = np.zeros((self.B,), dtype=bool)
        done_mask[env_id] = True
        self.env.reset(done=done_mask)

        # IMPORTANT:
        # 你当前 env 的 _reset_task_state(done) 是 return（不会清 task 状态）。
        # 为保证 episode 从头开始，这里手动清该 env 的任务变量（不改 env 文件）。
        try:
            self.env.current_btn_idx[env_id] = 0
            self.env.btn_pressed_mask[env_id, :] = False
            self.env.success_latched[env_id] = False
        except Exception:
            pass

        # lifecycle flags
        self.active[env_id] = True
        self.done[env_id] = False
        self.success[env_id] = False
        self.ctrl_step[env_id] = 0
        self.states[env_id] = self.ST_MOVE_ABOVE

        # initialize exec target from current obs
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
        info = self.env._state.info

        qpos = obs["qpos"][env_id].tolist()
        ee = obs["ee_pose"][env_id].tolist()
        grip = obs["gripper"][env_id].tolist()

        ctrl_vec = self._last_action[env_id].tolist()
        t_sec = float(self.ctrl_step[env_id] * float(self.env._cfg.ctrl_dt))

        # reuse env info (best-effort keys)
        cur_idx = int(np.asarray(info.get("cur_idx", np.array([0])))[env_id])
        completed = float(np.asarray(info.get("completed", np.zeros((self.B,), dtype=np.float32)))[env_id])
        dist = float(np.asarray(info.get("dist", np.zeros((self.B,), dtype=np.float32)))[env_id])
        touch_val = float(np.asarray(info.get("touch_val", np.zeros((self.B,), dtype=np.float32)))[env_id])
        is_success = bool(np.asarray(info.get("is_success", np.zeros((self.B,), dtype=np.bool_)))[env_id])

        rew = float(np.asarray(self.env._state.reward)[env_id])

        buf = self.buffers[env_id]
        buf["times"].append(t_sec)
        buf["logic_states"].append(int(self.states[env_id]))
        buf["qpos"].append(qpos)
        buf["ee_pose"].append(ee)
        buf["gripper"].append(grip)
        buf["ctrl"].append(ctrl_vec)

        buf["reward"].append(rew)
        buf["cur_idx"].append(cur_idx)
        buf["completed"].append(completed)
        buf["dist"].append(dist)
        buf["touch_val"].append(touch_val)
        buf["is_success"].append(is_success)

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
                    "reward": float(buf["reward"][i]),
                    "cur_idx": int(buf["cur_idx"][i]),
                    "completed": float(buf["completed"][i]),
                    "dist": float(buf["dist"][i]),
                    "touch_val": float(buf["touch_val"][i]),
                    "is_robot": True,
                    "logic_state": int(buf["logic_states"][i]),
                    "time": float(buf["times"][i]),
                    "success": bool(self.success[env_id]),
                    # 固定顺序（不随机）：0->1->2
                    "order": [0, 1, 2],
                    "is_success_env": bool(buf["is_success"][i]),
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

        # --- use env info BEFORE step to select current target ---
        info_pre = self.env._state.info
        cur_idx_pre = np.asarray(info_pre.get("cur_idx", np.zeros((B,), dtype=np.int32)), dtype=np.int32)
        cur_btn = np.clip(cur_idx_pre, 0, 2)  # target button index 0/1/2

        data = self.env._state.data

        # button poses (B,3,7) -> pos (B,3,3)
        btn_poses_list = [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.button_bodies]
        btn_poses = np.stack(btn_poses_list, axis=1)
        btn_pos = btn_poses[..., :3]

        cur_btn_pos = btn_pos[np.arange(B), cur_btn]  # (B,3)

        above_p = cur_btn_pos + np.asarray(cfg.above_offset, dtype=np.float32).reshape(1, 3)
        press_p = cur_btn_pos + np.asarray(cfg.press_offset, dtype=np.float32).reshape(1, 3)
        retreat_p = cur_btn_pos + np.asarray(cfg.retreat_offset, dtype=np.float32).reshape(1, 3)

        # choose target by FSM state
        tgt_pos = self.exec_pos.copy()
        s = self.states
        # print(s)

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

        # smooth to target
        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos, float(cfg.max_dp))

        # build action: [x,y,z,r,p,y,grip]
        action = np.zeros((B, 7), dtype=np.float32)
        action[:, :3] = self.exec_pos
        action[:, 3:6] = self.fixed_rpy
        action[:, 6] = float(cfg.gripper_open)

        self._last_action[:] = action

        # step env
        self.env.step(action)
        

        # --- reuse env info AFTER step ---
        info = self.env._state.info
        cur_idx_post = np.asarray(info.get("cur_idx", cur_idx_pre), dtype=np.int32)
        touch_val = np.asarray(info.get("touch_val", np.zeros((B,), dtype=np.float32)), dtype=np.float32)

        # reach checks (use exec_pos)
        reach_above = np.linalg.norm(self.exec_pos - above_p, axis=1) < float(cfg.pos_tol)
        reach_press = np.linalg.norm(self.exec_pos - press_p, axis=1) < float(cfg.pos_tol)
        reach_retreat = np.linalg.norm(self.exec_pos - retreat_p, axis=1) < float(cfg.pos_tol)

        # FSM transitions
        to_descend = running & (s == self.ST_MOVE_ABOVE) & reach_above
        if np.any(to_descend):
            self.states[to_descend] = self.ST_DESCEND

        to_hold = running & (s == self.ST_DESCEND) & reach_press
        if np.any(to_hold):
            self.states[to_hold] = self.ST_HOLD_PRESS

        # pressed detection: rely on env stage advance (cur_idx increases)
        advanced = running & (cur_idx_post > cur_idx_pre)

        # If advanced while holding/descending, retract
        to_retract = running & (self.states == self.ST_HOLD_PRESS) & advanced
        if np.any(to_retract):
            self.states[to_retract] = self.ST_RETRACT

        # (optional) if you want a fallback: retract when touch is high and at press pose
        # keep commented to ensure strict consistency with env
        # fallback_retract = running & (self.states == self.ST_HOLD_PRESS) & reach_press & (touch_val > float(cfg.touch_thresh))
        # if np.any(fallback_retract):
        #     self.states[fallback_retract] = self.ST_RETRACT

        to_next = running & (self.states == self.ST_RETRACT) & reach_retreat
        if np.any(to_next):
            # go back above for the (possibly new) target button
            self.states[to_next] = self.ST_MOVE_ABOVE

        # mark done if env reports success (reuse env is_success/terminated)
        is_success = np.asarray(info.get("is_success", np.zeros((B,), dtype=np.bool_)), dtype=np.bool_)
        done_by_env = running & is_success
        if np.any(done_by_env):
            self.states[done_by_env] = self.ST_DONE

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

            # terminal checks (reuse env info; no re-computation)
            info = self.env._state.info
            is_success = np.asarray(info.get("is_success", np.zeros((self.B,), dtype=np.bool_)), dtype=np.bool_)

            for i in range(self.B):
                if (not self.active[i]) or self.done[i]:
                    continue
                timeout = int(self.ctrl_step[i]) >= int(cfg.max_ctrl_steps)
                finished = bool(is_success[i]) or (int(self.states[i]) == self.ST_DONE)
                if finished or timeout:
                    self.done[i] = True
                    self.success[i] = bool(finished and (not timeout))

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
    p.add_argument("--action_mode", type=str, default=None, choices=["eef_relative", "eef", "joint"])
    p.add_argument("--max_ctrl_steps", type=int, default=None)
    args = p.parse_args()

    cfg = CollectorCfg(
        save_dir=args.save_dir if args.save_dir is not None else CollectorCfg.save_dir,
        data_size=args.data_size if args.data_size is not None else CollectorCfg.data_size,
        num_envs=args.num_envs if args.num_envs is not None else CollectorCfg.num_envs,
        seed=args.seed if args.seed is not None else CollectorCfg.seed,
        save_video=(not args.no_video),
        action_mode=args.action_mode if args.action_mode is not None else CollectorCfg.action_mode,
        max_ctrl_steps=args.max_ctrl_steps if args.max_ctrl_steps is not None else CollectorCfg.max_ctrl_steps,
    )

    runner = PressThreeButtonsCollector(cfg)
    try:
        runner.collect()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
