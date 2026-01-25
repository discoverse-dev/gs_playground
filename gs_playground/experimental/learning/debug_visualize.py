
import os
import sys
import numpy as np
import torch
import mujoco
from dataclasses import asdict
import mediapy as media
import math

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from gs_playground.src.locomotion.go2.walk_np import Go2WalkTaskMj
from gs_playground.src.locomotion.go2.cfg import Go2WalkNpEnvCfg
from gs_playground.src.env.mujoco_env.mj_env import MjNpEnv

def get_grid_offsets(num_envs, spacing=1.0):
    rows = int(math.ceil(math.sqrt(num_envs)))
    cols = int(math.ceil(num_envs / rows))
    offsets = np.zeros((num_envs, 2))
    for i in range(num_envs):
        r = i // cols
        c = i % cols
        offsets[i, 0] = r * spacing
        offsets[i, 1] = c * spacing
    return offsets

def render_many(model, data, state_batch, camera_name="track", shape=(320, 240), transparent=False, offsets=None):
    """
    Render multiple env states into one scene using mjv_addGeoms.
    state_batch: (num_envs, nstate) at a specific time step.
    """
    
    num_envs = state_batch.shape[0]
    
    # Visual options
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
    pert = mujoco.MjvPerturb()
    catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

    # Renderer
    # We use model (singular) for the renderer.
    renderer = mujoco.Renderer(model, height=shape[1], width=shape[0])
    
    # Function to set state
    def set_state(d, s, offset=None):
        # s is physics_state
        # time(1), qpos(nq), qvel(nv) ...
        d.time = s[0]
        d.qpos[:] = s[1:1+model.nq]
        d.qvel[:] = s[1+model.nq:1+model.nq+model.nv]
        if offset is not None:
            # Assumes the first 2 qpos are x, y which is true for free joint
            d.qpos[0] += offset[0]
            d.qpos[1] += offset[1]
        mujoco.mj_forward(model, d)

    # 1. Clear/Init Scene
    # Load state for robot 0 with its offset
    # We use robot 0 as the "main" one for scene init, but we want the camera to look at the group center.
    set_state(data, state_batch[0], offsets[0] if offsets is not None else None)
    
    # Init Camera
    cam = mujoco.MjvCamera()
    if offsets is not None:
        center_x = np.mean(offsets[:, 0])
        center_y = np.mean(offsets[:, 1])
        cam.lookat = [center_x, center_y, 0.0]
        cam.distance = 4.5
        cam.elevation = -20
        cam.azimuth = 135
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    else:
        pass

    # Update scene (creates the scene structure)
    renderer.update_scene(data, camera=cam, scene_option=vopt)
    
    # 2. Add other robots
    for i in range(1, num_envs):
        set_state(data, state_batch[i], offsets[i] if offsets is not None else None)
        mujoco.mjv_addGeoms(model, data, vopt, pert, catmask, renderer.scene)

    return renderer.render()

def main():
    # Config
    num_envs = 16
    max_steps = 100
    grid_spacing = 1.0
    video_fps = 25
    decimation = 2 
    
    cfg = Go2WalkNpEnvCfg()
    
    print(f"Initializing Env with {num_envs} envs...")
    env = Go2WalkTaskMj(cfg, num_envs=num_envs)
    env.init_state()
    
    print(f"Running rollout for {max_steps} steps...")
    
    state_history = []
    
    # Reset
    env.reset(np.arange(num_envs))
    
    for step in range(max_steps):
        # Actions - Use zero actions to test environment dynamics
        actions = np.zeros((num_envs, env.action_space.shape[0]))
        
        env.step(actions)
        
        # Check rewards
        # env.state.reward is (num_envs,)
        mean_reward = np.mean(env.state.reward)
        if step % 10 == 0:
            print(f"Step {step}: Mean Reward = {mean_reward:.4f}")
        
        # Store copy of physics state
        # state is (num_envs, nstate)
        state_history.append(env.state.physics_state.copy())

    print("Rendering...")
    
    # We reuse the env's worker data or create a new one for rendering
    # Single threaded render
    render_data = mujoco.MjData(env.model)
    
    offsets = get_grid_offsets(num_envs, spacing=grid_spacing)
    
    frames = []
    
    for i in range(0, len(state_history), decimation):
        print(f"Rendering frame {i}...", end="\r")
        
        frame = render_many(env.model, render_data, state_history[i], 
                            offsets=offsets, 
                            shape=(640, 480))
        frames.append(frame)
        
    print("\nSaving video...")
    media.write_video("debug_rollout.mp4", frames, fps=video_fps)
    print("Done. Saved to debug_rollout.mp4")

if __name__ == "__main__":
    main()
