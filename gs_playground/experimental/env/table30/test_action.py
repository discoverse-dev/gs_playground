import argparse
import math
import sys
from pathlib import Path

import numpy as np
import mediapy as media

# add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gs_playground.src.env import registry

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

def _get_robot_state(obs: dict, mode: str) -> np.ndarray:
    """Extract current robot state (joint or eef) from observations."""
    if "joint" in mode:
        qpos = np.asarray(obs["qpos"], dtype=np.float32) # (B, 7)
        if "gripper" in obs:
            gripper = np.asarray(obs["gripper"], dtype=np.float32).reshape(qpos.shape[0], -1)
            return np.concatenate([qpos, gripper], axis=1) # (B, 8)
        return qpos
    elif "eef" in mode:
        # eef_pose is (B, 6) [xyz, rpy]
        pose = np.asarray(obs["ee_pose"], dtype=np.float32)
        if "gripper" in obs:
             gripper = np.asarray(obs["gripper"], dtype=np.float32).reshape(pose.shape[0], -1)
             return np.concatenate([pose, gripper], axis=1) # (B, 7)
        return pose
    else:
        raise ValueError(f"Unknown mode type in {mode}")

def generate_targets(start_state: np.ndarray, mode: str) -> np.ndarray:
    """Generate a random target state based on start state."""
    B, D = start_state.shape
    target = start_state.copy()
    
    if "joint" in mode:
        # Move joints slightly (e.g. +/- 0.3 rad)
        # Avoid gripper (last dim) moving too much
        delta = np.random.uniform(-0.3, 0.3, size=(B, D))
        delta[:, -1] = 0.0 # Keep gripper steady
        target = start_state + delta
        
    elif "eef" in mode:
        # Move XYZ slightly (+/- 5cm), keep orientation roughly same or small rotation
        # XYZ
        target[:, :3] += np.random.uniform(-0.1, 0.1, size=(B, 3))
        # RPY (small wiggle)
        target[:, 3:6] += np.random.uniform(-0.1, 0.1, size=(B, 3))
        # Gripper
        target[:, -1] = np.random.choice([0.0, 0.5], size=(B,)) # Open or semi-closed
        
    return target

def run_test_mode(
    env_name: str,
    mode: str,
    output_dir: Path,
    num_envs: int = 4,
    steps: int = 100
):
    print(f"\n[Testing] Mode: {mode}")
    
    # Create env
    env = registry.make(
        env_name,
        sim_backend="np",
        env_cfg_override={"action_mode": mode},
        num_envs=num_envs,
    )
    
    # Reset
    obs, _ = env.reset()
    
    # Determine Start and Goal
    start_state = _get_robot_state(obs, mode)
    goal_state = generate_targets(start_state, mode)
    
    # Video writer
    video_path = output_dir / f"{mode}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    frames = []
    rgb0 = _extract_rgb(obs)
    grid_n = int(math.ceil(math.sqrt(num_envs)))
    
    if rgb0 is not None:
        grid = _make_grid(rgb0, grid_n)
        frames.append(grid)

    print(f"  Target generated. Moving {steps} steps...")
    
    # Delta step for 'delta' mode (constant velocity)
    delta_step = (goal_state - start_state) / steps

    for t in range(1, steps + 1):
        alpha = t / steps
        
        # Calculate ideal trajectory point at current time t
        traj_point = start_state + (goal_state - start_state) * alpha
        
        action = None
        
        if "absolute" in mode:
            # Action = Desired State
            action = traj_point
            
        elif "relative" in mode:
            # Action = Desired State - Reference State (Start)
            # Ref is fixed at chunk start (here we assume 1 chunk = whole episode for test)
            action = traj_point - start_state
            
        elif "delta" in mode:
            # Action = Step increment
            # target_t = target_{t-1} + action
            # We want linear motion, so constant action
            action = delta_step
            
        # Add shape noise? No, cleaner test
        
        # Step
        state = env.step(action)
        obs = state.obs

        # Visualize
        if frames is not None:
            rgb = _extract_rgb(obs)
            if rgb is not None:
                grid = _make_grid(rgb, grid_n)
                frames.append(grid)

    if frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        media.write_video(str(video_path), frames, fps=25.0)
        print(f"  Saved video to {video_path}")
        
    print("  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="table30/arrange_fruits")
    parser.add_argument("--output_dir", type=str, default="output/test_action")
    args = parser.parse_args()

    modes = [
        "joint_absolute",
        "joint_relative",
        "joint_delta",
        "eef_absolute",
        "eef_relative",
        "eef_delta"
    ]

    out_path = Path(args.output_dir)
    for m in modes:
        try:
            run_test_mode(args.env, m, out_path)
        except Exception as e:
            print(f"FAILED {m}: {e}")
            import traceback
            traceback.print_exc()
