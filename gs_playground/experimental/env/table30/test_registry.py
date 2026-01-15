import argparse
import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gs_playground.src.env import registry
from gs_playground.src.env.motrix_env.mtx_env import NpEnvState
from gs_playground.src.env.motrix_env.render_env import RenderEnvState


def _make_grid(rgb: np.ndarray, grid_n: int) -> np.ndarray:
    """Arrange (B,H,W,3) into grid_n x grid_n mosaic (pad black if not full)."""
    if rgb is None:
        return None
    B, H, W, C = rgb.shape
    rows = grid_n
    cols = grid_n
    canvas = np.zeros((rows * H, cols * W, C), dtype=np.uint8)
    for i in range(min(B, rows * cols)):
        r, c = divmod(i, cols)
        canvas[r * H : (r + 1) * H, c * W : (c + 1) * W] = rgb[i]
    return canvas


def _extract_rgb(obs: dict) -> np.ndarray:
    """Pick RGB batch from obs; prefer pixels/view_0."""
    view_keys = sorted([k for k in obs.keys() if k.startswith("pixels/view_")])
    if view_keys:
        return np.asarray(obs[view_keys[0]], dtype=np.uint8)
    return None


def _sample_targets(num_envs: int, obs: dict) -> np.ndarray:
    """
    Sample two random EE poses per env: start = current eef, goal = current eef + small delta.
    Returns (num_envs, 7) goal pose.
    """
    eef = np.asarray(obs.get("ee_pose", np.zeros((num_envs, 7), dtype=np.float32)), dtype=np.float32)
    if eef.ndim == 1:
        eef = np.tile(eef[None, :], (num_envs, 1))
    # random delta in xyz, keep orientation
    delta = np.zeros_like(eef)
    delta[:, :3] = np.random.uniform(low=-0.05, high=0.05, size=(num_envs, 3))
    goal = eef + delta
    return goal


def _interpolate(current: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    return current + (target - current) * alpha


def run_test(
    env_name: str,
    action_mode: str,
    num_envs: int,
    steps: int,
    save_video: bool,
    output: str,
    smooth_steps: int = 20,
):
    env = registry.make(
        env_name,
        sim_backend="np",
        env_cfg_override={"action_mode": action_mode},
        num_envs=num_envs,
    )
    reset_ret = env.reset()
    if isinstance(reset_ret, tuple):
        obs, info = reset_ret
    elif isinstance(reset_ret, (NpEnvState, RenderEnvState)):
        obs, info = reset_ret.obs, reset_ret.info
    else:
        raise RuntimeError(f"Unexpected reset return type: {type(reset_ret)}")
    print(f"[reset] env={env_name}, action_mode={action_mode}, num_envs={num_envs}")

    video_path = Path(output) / env_name / f"{action_mode}_mode.mp4"
    writer = None
    grid_n = int(math.ceil(math.sqrt(num_envs)))
    if save_video:
        rgb0 = _extract_rgb(obs)
        if rgb0 is not None:
            grid_frame = _make_grid(rgb0, grid_n)
            if grid_frame is not None:
                Path(video_path).parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                h, w = grid_frame.shape[:2]
                writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (w, h))
                writer.write(grid_frame[..., ::-1])  # RGB -> BGR

    # Targets setup
    goal_pose = None
    start_joints, target_joints = None, None
    
    if action_mode == "eef":
        # sample two targets per env: start=reset eef, goal=reset eef+delta; then interpolate across steps
        goal_pose = _sample_targets(num_envs, obs)
    
    elif action_mode == "joint":
        # prepare joint limits for sampling
        space = env.action_space
        # Ensure we have (num_envs, D) unique targets
        # Gym Box space low/high are typically (D,)
        
        # construct current start config (qpos + gripper)
        qpos_start = np.asarray(obs["qpos"], dtype=np.float32) # (B, 7)
        if "gripper" in obs:
            grp_start = np.asarray(obs["gripper"], dtype=np.float32).reshape(num_envs, -1)
            start_joints = np.concatenate([qpos_start, grp_start], axis=1) # (B, 8)
        else:
            start_joints = qpos_start
            
        # sample target config
        # Explicitly pass size to ensure broadcasting happens in sampling, not result
        dim = space.shape[0]
        target_joints = np.random.uniform(low=space.low, high=space.high, size=(num_envs, dim)).astype(np.float32)

    # Logging containers for Env 0
    hist_qpos_0 = []
    hist_action_0 = []

    for t in range(steps):
        alpha = min(1.0, (t + 1) / max(1, smooth_steps))
        
        act = np.zeros((num_envs, 8), dtype=np.float32) # Default buffer

        if action_mode == "eef":
            if "ee_pose" in obs:
                cur = np.asarray(obs["ee_pose"], dtype=np.float32)
                interp = _interpolate(cur, goal_pose, alpha)
                act[:, :7] = interp
            else:
                act[:, :7] = 0.0
        elif action_mode == "joint":
            # joint space: interpolate start -> target
            act = _interpolate(start_joints, target_joints, alpha)
        
        # Clip action to be safe (though interpolate should be safe if endpoints are safe)
        if action_mode == "joint":
             space = env.action_space
             low = np.asarray(space.low, dtype=np.float32).reshape(num_envs, -1) if np.isscalar(space.low) else space.low
             high = np.asarray(space.high, dtype=np.float32).reshape(num_envs, -1) if np.isscalar(space.high) else space.high
             act = np.clip(act, low, high)

        step_ret = env.step(act)        
        state = step_ret
        obs = state.obs
        reward = state.reward
        terminated = state.terminated
        truncated = state.truncated
        info = state.info

        # Log for Env 0
        if action_mode == "joint":
            # Actual joints
            q = np.asarray(obs["qpos"], dtype=np.float32)[0] # (7,) or (8,) depending on env def, usually 7
            if "gripper" in obs:
                g = np.asarray(obs["gripper"], dtype=np.float32)[0]
                q = np.concatenate([q, g])
            hist_qpos_0.append(q)
            hist_action_0.append(act[0].copy())

        if writer is not None:
            rgb_b = _extract_rgb(obs)
            if rgb_b is not None:
                grid_frame = _make_grid(rgb_b, grid_n)
                if grid_frame is not None:
                    writer.write(grid_frame[..., ::-1])

        print(f"[step {t+1}] reward_mean={np.mean(reward):.4f}, success={np.sum(info.get('is_success', 0))}")
        if np.all(terminated | truncated):
            print(f"[done early at {t+1}] all envs terminated or truncated")
            break

    if writer is not None:
        writer.release()
        print(f"[video] saved to {video_path}")
    
    # Plotting for Joint mode
    if action_mode == "joint" and len(hist_qpos_0) > 0:
        hist_qpos_0 = np.array(hist_qpos_0)   # (T, D)
        hist_action_0 = np.array(hist_action_0) # (T, D)
        
        out_dir = Path(video_path).parent
        
        # Plot 1: Actual Joint Angles
        plt.figure(figsize=(10, 6))
        for i in range(hist_qpos_0.shape[1]):
            plt.plot(hist_qpos_0[:, i], label=f"J{i}")
        plt.title("Env 0: Actual Joint Angles (qpos + gripper)")
        plt.xlabel("Step")
        plt.ylabel("Angle / Pos")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(out_dir / "env0_joint_actual.png")
        plt.close()
        print(f"[plot] saved {out_dir / 'env0_joint_actual.png'}")

        # Plot 2: Command Joint Actions
        plt.figure(figsize=(10, 6))
        for i in range(hist_action_0.shape[1]):
            plt.plot(hist_action_0[:, i], linestyle="--", label=f"Cmd{i}")
        plt.title("Env 0: Commanded Joint Actions")
        plt.xlabel("Step")
        plt.ylabel("Command")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(out_dir / "env0_joint_command.png")
        plt.close()
        print(f"[plot] saved {out_dir / 'env0_joint_command.png'}")

    print("[done] test finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="table30/stack_color_blocks")
    parser.add_argument("--action_mode", type=str, default="joint", choices=["joint", "eef"])
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--smooth_steps", type=int, default=20, help="steps to reach target pose")
    parser.add_argument("--no_video", action="store_true", help="disable video saving")
    parser.add_argument("--output", type=str, default="output/test_registry")
    args = parser.parse_args()

    run_test(
        env_name=args.env,
        action_mode=args.action_mode,
        num_envs=args.num_envs,
        steps=args.steps,
        save_video=not args.no_video,
        output=args.output,
        smooth_steps=args.smooth_steps,
    )
