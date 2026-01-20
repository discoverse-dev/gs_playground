from __future__ import annotations

import os
import json
import time
import shutil
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

from gs_playground.src.manipulation.tasks.table30._03_arrange_fruits_in_basket import (
    ArrangeFruitsEnv,
    ArrangeFruitsEnvCfg,
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
# Offsets
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class StageOffsets:
    above_obj: Tuple[float, float, float]
    grasp: Tuple[float, float, float]
    lift: Tuple[float, float, float]
    above_container: Tuple[float, float, float]


# -----------------------------------------------------------------------------
# Config (collector-only; env 已有的不要重复定义)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CollectorCfg:
    # dataset
    data_size: int = 1
    num_envs: int = 1
    seed: int = 0
    save_dir: str = "./data/table30_arrange_fruits_collect_1k_5"

    # env control（只保留 collector 的“上限”）
    max_ctrl_steps: int = 1200

    # motion
    max_dp: float = 0.005
    pos_tol: float = 0.012

    # gripper (match your runner convention)
    gripper_open: float = 0.0
    gripper_close: float = 0.62
    close_hold_steps: int = 25
    release_hold_steps: int = 25

    # sampling / render
    sample_every_steps: int = 1
    save_video: bool = True
    render_every_steps: int = 1
    video_fps: int = 30
    video_w: int = 320
    video_h: int = 240
    cam_view_key: Optional[str] = None  # None -> 默认 "pixels/view_0"

    # MotrixSim 渲染器视频录制配置
    enable_motrix_video: bool = True  # 启用 MotrixSim 渲染器录制
    motrix_video_fps: int = 30
    motrix_video_width: int = 320
    motrix_video_height: int = 240
    # Note: MotrixSim videos are saved to the same directory as 3DGS videos ({save_dir}/videos)

    # text fields（若 None，则默认继承 env cfg）
    subtask: Optional[str] = None  # None -> env._cfg.instruction
    prompt: Optional[str] = None  # None -> 从 instruction 简单派生

    # offsets
    default_offsets: StageOffsets = StageOffsets(
        above_obj=(0.0, 0.0, 0.15),
        grasp=(0.0, 0.0, 0.01),
        lift=(0.0, 0.0, 0.20),
        above_container=(0.0, 0.0, 0.10),
    )

    obj_offsets: Dict[str, StageOffsets] = field(
        default_factory=lambda: {
            "fruit_avocado": StageOffsets(
                above_obj=(0.0, 0.0, 0.10),
                grasp=(0.0, -0.05, -0.015),
                lift=(0.0, 0.0, 0.20),
                above_container=(0.0, 0.0, 0.10),
            ),
            "fruit_banana": StageOffsets(
                above_obj=(0.0, -0.05, 0.10),
                grasp=(0.0, -0.05, -0.02),
                lift=(0.0, 0.0, 0.20),
                above_container=(0.0, 0.0, 0.10),
            ),
            "fruit_carambola": StageOffsets(
                above_obj=(-0.02, -0.03, 0.20),
                grasp=(-0.02, -0.03, -0.02),
                lift=(0.0, 0.0, 0.20),
                above_container=(0.0, 0.0, 0.10),
            ),
            "fruit_mangosteen": StageOffsets(
                above_obj=(-0.03, 0.0, 0.20),
                grasp=(-0.03, 0, -0.07),
                lift=(0.0, 0.0, 0.20),
                above_container=(0.0, 0.0, 0.15),
            ),
        }
    )


# -----------------------------------------------------------------------------
# Collector (success-only JSONL + MP4)
# -----------------------------------------------------------------------------
class ArrangeFruitsCollector:
    ST_IDLE = 0
    ST_GO_ABOVE = 1
    ST_DESCEND = 2
    ST_CLOSE = 3
    ST_LIFT = 4
    ST_GO_BASKET = 5
    ST_OPEN = 6
    ST_RETREAT = 7
    ST_DONE = 8

    def __init__(
        self, cfg: CollectorCfg, env_cfg: Optional[ArrangeFruitsEnvCfg] = None
    ):
        self.cfg = cfg
        os.makedirs(cfg.save_dir, exist_ok=True)
        self.videos_dir = os.path.join(cfg.save_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

        # env cfg：优先外部传入，否则用默认
        self.env_cfg = env_cfg if env_cfg is not None else ArrangeFruitsEnvCfg()

        # 传递 MotrixSim 视频配置到环境配置
        self.env_cfg.enable_motrix_video = cfg.enable_motrix_video
        self.env_cfg.motrix_video_fps = cfg.motrix_video_fps
        self.env_cfg.motrix_video_width = cfg.motrix_video_width
        self.env_cfg.motrix_video_height = cfg.motrix_video_height
        # MotrixSim 视频保存到与 3DGS 视频相同的目录
        self.env_cfg.motrix_video_output_dir = self.videos_dir

        self.env = ArrangeFruitsEnv(self.env_cfg, num_envs=int(cfg.num_envs))
        self.env.reset()

        self.model = self.env.model
        self.B = int(cfg.num_envs)

        # cam view key
        self.cam_view_key = cfg.cam_view_key or "pixels/view_0"

        # episode prompt/subtask：默认继承 env
        instruction = str(getattr(self.env._cfg, "instruction", "") or "")
        self.ep_subtask = np.array([cfg.subtask or instruction] * self.B, dtype=object)
        self.ep_prompt = np.array(
            [cfg.prompt or (instruction if instruction else "arrange fruits")] * self.B,
            dtype=object,
        )

        # per-env lifecycle
        B = self.B
        self.active = np.zeros(B, dtype=bool)
        self.done = np.zeros(B, dtype=bool)
        self.success = np.zeros(B, dtype=bool)
        self.ctrl_step = np.zeros(B, dtype=np.int32)
        self._attempt_id = np.zeros(B, dtype=np.int64)

        # FSM + dwell
        self.states = np.zeros(B, dtype=np.int32)
        self.state_enter_step = np.zeros(B, dtype=np.int32)

        # fixed rpy per episode (来自 env obs)
        self.fixed_rpy = np.zeros((B, 3), dtype=np.float32)

        # exec target pos
        self.exec_pos = np.zeros((B, 3), dtype=np.float32)

        # logic state
        self.current_fruit_idx = np.zeros(B, dtype=np.int32)
        self.latched_fruit_pos = np.zeros((B, 3), dtype=np.float32)
        self.latched_basket_pos = np.zeros((B, 3), dtype=np.float32)

        # buffers for jsonl
        self.buffers: List[Dict[str, Any]] = [self._new_buffer() for _ in range(B)]

        # per-env video writer
        self.video_writers: List[Optional[EpisodeVideoWriter]] = [None] * B
        self._tmp_video_paths: List[str] = [
            os.path.join(self.videos_dir, f"_tmp_env{i}.mp4") for i in range(B)
        ]

        # stats
        self.saved_success = 0
        self.attempted = 0
        self._last_log_t = time.perf_counter()

        # cached last action for ctrl logging
        self._last_action = np.zeros((B, 7), dtype=np.float32)

        # handles
        self.basket_site = self.env.basket_site
        self.fruit_bodies = self.env.fruit_bodies

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
            # env info (best-effort)
            "cur_idx_env": [],
            "completed": [],
            "is_grasped_env": [],
            "touch_val": [],
            "dist_ee_fruit": [],
            "dist_fruit_basket": [],
            "in_basket": [],
            "is_success": [],
            "video_frames": 0,
        }

    # ----------------------------
    # Small helpers: robust info reading
    # ----------------------------
    def _info_get_scalar(
        self, info: Dict[str, Any], env_id: int, keys: List[str], default: float = 0.0
    ) -> float:
        for k in keys:
            if k in info and info[k] is not None:
                a = np.asarray(info[k]).reshape(-1)
                if a.size > env_id:
                    return float(a[env_id])
        return float(default)

    def _info_get_bool(
        self, info: Dict[str, Any], env_id: int, keys: List[str], default: bool = False
    ) -> bool:
        for k in keys:
            if k in info and info[k] is not None:
                a = np.asarray(info[k]).reshape(-1)
                if a.size > env_id:
                    return bool(a[env_id])
        return bool(default)

    def _info_get_int(
        self, info: Dict[str, Any], env_id: int, keys: List[str], default: int = 0
    ) -> int:
        for k in keys:
            if k in info and info[k] is not None:
                a = np.asarray(info[k]).reshape(-1)
                if a.size > env_id:
                    return int(a[env_id])
        return int(default)

    def _get_stage_offsets(self, fruit_name: str) -> StageOffsets:
        return self.cfg.obj_offsets.get(str(fruit_name), self.cfg.default_offsets)

    def start_episodes(self, env_ids: np.ndarray, seed: int) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if env_ids.size == 0:
            return

        # Set env RNG once for this batch reset

        self.env._rng = np.random.default_rng(int(seed))
        self.env.robot.update_reference(self.env._state.data)

        done_mask = np.zeros((self.B,), dtype=bool)
        done_mask[env_ids] = True
        self.env.reset(done=done_mask)

        self.active[env_ids] = True
        self.done[env_ids] = False
        self.success[env_ids] = False
        self.ctrl_step[env_ids] = 0
        self.states[env_ids] = self.ST_IDLE
        self.state_enter_step[env_ids] = 0

        # init exec_pos + fixed_rpy from obs
        obs = self.env._state.obs
        ee6_all = np.asarray(obs["ee_pose"], dtype=np.float32).reshape(
            self.B, -1
        )  # (B,6)
        self.exec_pos[env_ids] = ee6_all[env_ids, :3]
        self.fixed_rpy[env_ids] = ee6_all[env_ids, 3:6]

        # init fruit idx
        self.current_fruit_idx[env_ids] = 0

        # latch basket pos once per episode start (can also refresh per step if you prefer)
        data = self.env._state.data
        basket_pose7 = np.asarray(
            self.basket_site.get_pose(data), dtype=np.float32
        ).reshape(self.B, -1)
        self.latched_basket_pos[env_ids] = basket_pose7[env_ids, :3]

        # buffers/video reset
        for env_id in env_ids.tolist():
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
                    tmp_path,
                    int(self.cfg.video_fps),
                    (int(self.cfg.video_w), int(self.cfg.video_h)),
                )

            # Restart MotrixSim video recorder for env_id=0
            # (MotrixSim only records the first environment)
            if env_id == 0 and self.cfg.enable_motrix_video:
                if (
                    hasattr(self.env, "_motrix_recorder")
                    and self.env._motrix_recorder is not None
                ):
                    # Use next saved_success as episode index for naming
                    ep_idx = int(self.saved_success)
                    self.env._motrix_recorder.restart_episode(self.videos_dir, ep_idx)

            self._attempt_id[env_id] += 1
            self.attempted += 1

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
        ctrl_dt = float(getattr(self.env._cfg, "ctrl_dt", 0.02))
        t_sec = float(self.ctrl_step[env_id] * ctrl_dt)

        rew = float(np.asarray(self.env._state.reward).reshape(-1)[env_id])

        # env metrics (best effort; keys aligned with your ArrangeFruitsEnv)
        cur_idx_env = self._info_get_int(info, env_id, ["cur_idx", "cur_target_idx"], 0)
        completed = self._info_get_int(info, env_id, ["completed"], 0)
        is_grasped_env = self._info_get_bool(info, env_id, ["is_grasped"], False)
        touch_val = self._info_get_scalar(info, env_id, ["touch_val", "touch"], 0.0)
        d_ee_fruit = self._info_get_scalar(
            info, env_id, ["dist_ee_fruit", "d_ee_fruit"], 0.0
        )
        d_fruit_basket = self._info_get_scalar(
            info, env_id, ["dist_fruit_basket", "d_fruit_basket"], 0.0
        )
        in_basket = self._info_get_bool(info, env_id, ["in_basket"], False)
        is_success = self._info_get_bool(info, env_id, ["is_success", "success"], False)

        buf = self.buffers[env_id]
        buf["times"].append(t_sec)
        buf["logic_states"].append(int(self.states[env_id]))
        buf["qpos"].append(qpos)
        buf["ee_pose"].append(ee)
        buf["gripper"].append(grip)
        buf["ctrl"].append(ctrl_vec)
        buf["reward"].append(rew)

        buf["cur_idx_env"].append(cur_idx_env)
        buf["completed"].append(completed)
        buf["is_grasped_env"].append(is_grasped_env)
        buf["touch_val"].append(touch_val)
        buf["dist_ee_fruit"].append(d_ee_fruit)
        buf["dist_fruit_basket"].append(d_fruit_basket)
        buf["in_basket"].append(in_basket)
        buf["is_success"].append(is_success)

    def _write_video_frame(self, env_id: int) -> None:
        vw = self.video_writers[env_id]
        if vw is None:
            return
        obs = self.env._state.obs
        if self.cam_view_key not in obs:
            return
        rgb = obs[self.cam_view_key][env_id]
        if rgb is None:
            return
        bgr = rgb[..., ::-1].copy()
        vw.write(bgr)
        self.buffers[env_id]["video_frames"] += 1

    def _flush_episode_jsonl(
        self, env_id: int, ep_idx: int, video_rel_path: str
    ) -> None:
        path = os.path.join(self.cfg.save_dir, f"episode_{ep_idx:05d}.jsonl")
        buf = self.buffers[env_id]
        n = len(buf["times"])

        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                legacy_state = buf["qpos"][i] + buf["gripper"][i]
                rec = {
                    "images_1": {
                        "url": video_rel_path,
                        "type": "video",
                        "frame_idx": i,
                    },
                    "prompt": str(self.ep_prompt[env_id]),
                    "state": legacy_state,
                    "qpos": buf["qpos"][i],
                    "ee_pose": buf["ee_pose"][i],
                    "ctrl": buf["ctrl"][i],
                    "gripper": buf["gripper"][i],
                }
                # optional: write latches only once
                if i == 0:
                    rec["latched_basket_pos"] = self.latched_basket_pos[env_id].tolist()
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _finalize_episode(self, env_id: int) -> None:
        if self.video_writers[env_id] is not None:
            self.video_writers[env_id].close()
            self.video_writers[env_id] = None

        tmp_path = self._tmp_video_paths[env_id]

        if self.success[env_id] and (self.saved_success < int(self.cfg.data_size)):
            # if True:
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
    def _enter_state(self, mask: np.ndarray, new_state: int) -> None:
        if not np.any(mask):
            return
        self.states[mask] = int(new_state)
        self.state_enter_step[mask] = self.ctrl_step[mask].copy()

    def _step_logic(self) -> None:
        cfg = self.cfg
        B = self.B

        running = self.active & (~self.done)
        if not np.any(running):
            return

        data = self.env._state.data

        # Sync our current fruit idx with env internal index (env increments when place succeeds)
        env_cur = np.asarray(
            getattr(self.env, "current_obj_idx", np.zeros((B,), dtype=np.int32))
        ).reshape(-1)
        N = len(self.env._cfg.fruit_names)
        advance = running & (env_cur > self.current_fruit_idx) & (env_cur < N)
        if np.any(advance):
            self.current_fruit_idx[advance] = env_cur[advance]
            self.states[advance] = self.ST_IDLE
            self.state_enter_step[advance] = self.ctrl_step[advance].copy()

        # If env says done (all fruits placed), mark done
        done_all = running & (env_cur >= len(self.env._cfg.fruit_names))
        if np.any(done_all):
            self.states[done_all] = self.ST_DONE

        # Latch fruit pos at IDLE -> GO_ABOVE
        m_idle = (
            running
            & (self.states == self.ST_IDLE)
            & (self.current_fruit_idx < len(self.env._cfg.fruit_names))
        )
        if np.any(m_idle):
            idxs = self.current_fruit_idx.copy()
            # latch current fruit pose
            for i in np.where(m_idle)[0].tolist():
                f_i = int(idxs[i])
                pose7 = np.asarray(
                    self.fruit_bodies[f_i].get_pose(data[i]), dtype=np.float32
                ).reshape(-1)
                self.latched_fruit_pos[i] = pose7[:3]
            self._enter_state(m_idle, self.ST_GO_ABOVE)

        # Build targets
        tgt_pos = self.exec_pos.copy()
        grip_cmd = np.full((B,), float(cfg.gripper_open), dtype=np.float32)

        # per-env stage points (computed from latched fruit + basket + offsets)
        above_p = np.zeros((B, 3), dtype=np.float32)
        grasp_p = np.zeros((B, 3), dtype=np.float32)
        lift_p = np.zeros((B, 3), dtype=np.float32)
        basket_p = np.zeros((B, 3), dtype=np.float32)
        retreat_p = np.zeros((B, 3), dtype=np.float32)

        fruit_names = self.env._cfg.fruit_names
        for i in np.where(running)[0].tolist():
            cur_i = int(self.current_fruit_idx[i])
            if cur_i >= len(fruit_names):
                continue
            name = str(fruit_names[cur_i])
            offs = self._get_stage_offsets(name)

            fp = self.latched_fruit_pos[i]
            bp = self.latched_basket_pos[i]

            above_p[i] = fp + np.asarray(offs.above_obj, dtype=np.float32)
            grasp_p[i] = fp + np.asarray(offs.grasp, dtype=np.float32)
            lift_p[i] = grasp_p[i] + np.asarray(offs.lift, dtype=np.float32)
            basket_p[i] = bp + np.asarray(offs.above_container, dtype=np.float32)
            retreat_p[i] = basket_p[i].copy()
            retreat_p[i, 2] += 0.2

        s = self.states
        m1 = running & (s == self.ST_GO_ABOVE)
        m2 = running & (s == self.ST_DESCEND)
        m3 = running & (s == self.ST_CLOSE)
        m4 = running & (s == self.ST_LIFT)
        m5 = running & (s == self.ST_GO_BASKET)
        m6 = running & (s == self.ST_OPEN)
        m7 = running & (s == self.ST_RETREAT)

        if np.any(m1):
            tgt_pos[m1] = above_p[m1]
            grip_cmd[m1] = float(cfg.gripper_open)

        if np.any(m2):
            tgt_pos[m2] = grasp_p[m2]
            grip_cmd[m2] = float(cfg.gripper_open)

        if np.any(m3):
            tgt_pos[m3] = grasp_p[m3]
            grip_cmd[m3] = float(cfg.gripper_close)

        if np.any(m4):
            tgt_pos[m4] = lift_p[m4]
            grip_cmd[m4] = float(cfg.gripper_close)

        if np.any(m5):
            tgt_pos[m5] = basket_p[m5]
            grip_cmd[m5] = float(cfg.gripper_close)

        if np.any(m6):
            tgt_pos[m6] = basket_p[m6]
            grip_cmd[m6] = float(cfg.gripper_open)

        if np.any(m7):
            tgt_pos[m7] = retreat_p[m7]
            grip_cmd[m7] = float(cfg.gripper_open)

        ref_pose_6d = self.env.robot.ref_ee_pose  # (B,6)

        ref_pos = ref_pose_6d[:, :3]
        ref_rpy = ref_pose_6d[:, 3:6]

        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos, float(cfg.max_dp))

        action = np.zeros((B, 7), dtype=np.float32)
        action[:, :3] = self.exec_pos - ref_pos
        action[:, :2] *= 0.5
        action[:, 2] *= 1
        action[:, 3:6] = 0
        action[:, 6] = grip_cmd
        self._last_action[:] = action

        self.env.step(action)

        # reach checks (use exec_pos)
        def _reach(p: np.ndarray) -> np.ndarray:
            return np.linalg.norm(self.exec_pos - p, axis=1) < float(cfg.pos_tol)

        reach_above = _reach(above_p)
        reach_grasp = _reach(grasp_p)
        reach_lift = _reach(lift_p)
        reach_basket = _reach(basket_p)
        reach_retreat = _reach(retreat_p)

        self._enter_state(
            running & (s == self.ST_GO_ABOVE) & reach_above, self.ST_DESCEND
        )
        self._enter_state(running & (s == self.ST_DESCEND) & reach_grasp, self.ST_CLOSE)

        in_close = running & (self.states == self.ST_CLOSE)
        close_done = in_close & (
            (self.ctrl_step - self.state_enter_step) >= int(cfg.close_hold_steps)
        )
        self._enter_state(close_done, self.ST_LIFT)

        self._enter_state(
            running & (self.states == self.ST_LIFT) & reach_lift, self.ST_GO_BASKET
        )
        self._enter_state(
            running & (self.states == self.ST_GO_BASKET) & reach_basket, self.ST_OPEN
        )

        in_open = running & (self.states == self.ST_OPEN)

        open_done = in_open & (
            (self.ctrl_step - self.state_enter_step) >= int(cfg.release_hold_steps)
        )
        self._enter_state(open_done, self.ST_RETREAT)

        self._enter_state(
            running & (self.states == self.ST_RETREAT) & reach_retreat, self.ST_IDLE
        )

        # done by env success latch（兼容 is_success / success）
        info = self.env._state.info
        is_success = np.asarray(
            info.get("is_success", info.get("success", np.zeros((B,), dtype=np.bool_))),
            dtype=np.bool_,
        ).reshape(-1)
        done_by_env = running & is_success
        if np.any(done_by_env):
            self.states[done_by_env] = self.ST_DONE

    # ----------------------------
    # Collect loop
    # ----------------------------
    def collect(self) -> None:
        cfg = self.cfg
        target = int(cfg.data_size)

        all_ids = np.arange(self.B, dtype=np.int64)
        self.start_episodes(all_ids, seed=int(cfg.seed))

        while self.saved_success < target:
            # 1) step env + update state machine
            self._step_logic()

            running = self.active & (~self.done)

            # 2) capture
            sample_mask = running & (
                (self.ctrl_step % int(cfg.sample_every_steps)) == 0
            )
            for env_id in np.where(sample_mask)[0].tolist():
                self._capture_step(int(env_id))

            # 3) render
            if cfg.save_video:
                render_mask = running & (
                    (self.ctrl_step % int(cfg.render_every_steps)) == 0
                )
                for env_id in np.where(render_mask)[0].tolist():
                    self._write_video_frame(int(env_id))

            # 4) increment ctrl steps for running envs
            self.ctrl_step[running] += 1

            # 5) read env success
            info = self.env._state.info
            is_success = np.asarray(
                info.get(
                    "is_success",
                    info.get("success", np.zeros((self.B,), dtype=np.bool_)),
                ),
                dtype=np.bool_,
            ).reshape(-1)

            # 6) mark done/success (per-env)
            for i in range(self.B):
                if (not self.active[i]) or self.done[i]:
                    continue

                timeout = int(self.ctrl_step[i]) >= int(cfg.max_ctrl_steps)
                finished = bool(is_success[i]) or (
                    int(self.states[i]) == int(self.ST_DONE)
                )

                if finished or timeout:
                    self.done[i] = True
                    self.success[i] = bool(is_success[i])

            # 7) finalize ALL done envs first (no restart inside per-env loop!)
            done_ids = np.where(self.active & self.done)[0]
            for env_id in done_ids.tolist():
                self._finalize_episode(int(env_id))

            # 8) if reached target, stop remaining envs and exit loop naturally
            if self.saved_success >= target:
                self.active[done_ids] = False
            else:
                # 9) restart done envs in one batch (only those still active)
                restart_ids = done_ids[self.active[done_ids]]
                if restart_ids.size > 0:
                    batch_seed = int(cfg.seed + self.attempted)
                    self.done[restart_ids] = False
                    self.start_episodes(restart_ids, seed=batch_seed)

            # 10) periodic log
            now = time.perf_counter()
            if (now - self._last_log_t) >= 2.0:
                print(
                    f"[collect] active={int(np.sum(self.active))}/{self.B} "
                    f"saved_success={int(self.saved_success)}/{target} attempted={int(self.attempted)}"
                )
                self._last_log_t = now

        print(
            f"[DONE] saved_success={self.saved_success}/{target}, attempted={int(self.attempted)}"
        )
        print(f"Saved to: {cfg.save_dir}")

    def close(self) -> None:
        # 关闭 MotrixSim 视频录制器
        if hasattr(self.env, "close"):
            self.env.close()

        # 关闭 3DGS 视频写入器
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

    # MotrixSim 视频录制参数
    p.add_argument(
        "--enable_motrix_video",
        action="store_true",
        default=True,
        help="Enable MotrixSim renderer video recording (default: False)",
    )

    # env cfg override（只允许改 env 已有字段）
    p.add_argument(
        "--action_mode",
        type=str,
        default="eef_relative",
        choices=["eef", "eef_relative", "joint"],
    )
    args = p.parse_args()

    env_cfg = ArrangeFruitsEnvCfg()
    if args.action_mode is not None:
        env_cfg.action_mode = str(args.action_mode)

    cfg = CollectorCfg(
        save_dir=args.save_dir if args.save_dir is not None else CollectorCfg.save_dir,
        data_size=args.data_size
        if args.data_size is not None
        else CollectorCfg.data_size,
        num_envs=args.num_envs if args.num_envs is not None else CollectorCfg.num_envs,
        seed=args.seed if args.seed is not None else CollectorCfg.seed,
        save_video=(not args.no_video),
        max_ctrl_steps=args.max_ctrl_steps
        if args.max_ctrl_steps is not None
        else CollectorCfg.max_ctrl_steps,
        # MotrixSim 视频录制配置
        enable_motrix_video=(args.enable_motrix_video),
    )

    runner = ArrangeFruitsCollector(cfg, env_cfg=env_cfg)
    try:
        runner.collect()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
