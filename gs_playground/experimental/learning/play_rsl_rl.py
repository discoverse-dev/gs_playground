
import os
import sys
import numpy as np
import torch
import mujoco
from dataclasses import asdict
import mediapy as media
import math
import glob

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from gs_playground.src.locomotion.go2.walk_np import Go2WalkTaskMj
from gs_playground.src.locomotion.go2.cfg import Go2WalkNpEnvCfg
from gs_playground.experimental.learning.train_rsl_rl import RslMjEnvWrapper
from rsl_rl.runners import OnPolicyRunner

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

def render_many(model, data, state_batch, shape=(640, 480), transparent=False, offsets=None):
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
        # Go2 model: nq=19 (7 base + 12 joints), nv=18 (6 base + 12 joints)
        d.time = s[0]
        d.qpos[:] = s[1:1+model.nq]
        d.qvel[:] = s[1+model.nq:1+model.nq+model.nv]
        if offset is not None:
            # Assumes the first 2 qpos are x, y (free joint)
            d.qpos[0] += offset[0]
            d.qpos[1] += offset[1]
        mujoco.mj_forward(model, d)

    # 1. Clear/Init Scene
    # Load state for robot 0 with its offset
    set_state(data, state_batch[0], offsets[0] if offsets is not None else None)
    
    # Init Camera
    cam = mujoco.MjvCamera()
    if offsets is not None:
        center_x = np.mean(offsets[:, 0])
        center_y = np.mean(offsets[:, 1])
        cam.lookat = [center_x, center_y, 0.0]
        cam.distance = 6.0
        cam.elevation = -45
        cam.azimuth = 90
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    # Update scene (creates the scene structure)
    renderer.update_scene(data, camera=cam, scene_option=vopt)
    
    # 2. Add other robots
    for i in range(1, num_envs):
        set_state(data, state_batch[i], offsets[i] if offsets is not None else None)
        mujoco.mjv_addGeoms(model, data, vopt, pert, catmask, renderer.scene)

    return renderer.render()

def find_latest_model(log_dir):
    # Pattern: model_*.pt
    files = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if not files:
        return None
    
    # Sort by iteration number
    def extract_iter(f):
        base = os.path.basename(f)
        try:
            return int(base.split('_')[1].split('.')[0])
        except:
            return -1
            
    files.sort(key=extract_iter, reverse=True)
    return files[0]

def main():
    # Config
    num_envs = 16 # Visualizing 16 dogs
    max_steps = 300 # 300 steps
    grid_spacing = 1.0
    video_fps = 25
    decimation = 2 # Render every 2nd step (50Hz control -> 25fps)
    
    # Paths
    logs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../logs"))
    model_path = find_latest_model(logs_root)
    
    if not model_path:
        print(f"No model found in {logs_root}")
        return
        
    print(f"Loading model: {model_path}")
    
    # 1. Environment
    env_cfg = Go2WalkNpEnvCfg()
    
    print(f"Initializing Env with {num_envs} envs...")
    env = Go2WalkTaskMj(env_cfg, num_envs=num_envs)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vec_env = RslMjEnvWrapper(env, device)
    
    # 2. Runner & Policy
    train_cfg = asdict(env_cfg.train_cfg)
    if "runner" in train_cfg:
        train_cfg.update(train_cfg.pop("runner"))
        
    # Initialize runner structure
    runner = OnPolicyRunner(vec_env, train_cfg, log_dir=logs_root, device=device)
    
    # Load model weights
    loaded_dict = runner.load(model_path, load_optimizer=False)
    if loaded_dict is not None:
        print(f"Loaded iteration: {loaded_dict.get('iter', 'unknown')}")
    else:
        print("Model loaded (no dict returned).")
    
    policy = runner.alg.policy.actor
    policy.eval()
    
    # 3. Simulate
    print(f"Running play for {max_steps} steps...")
    
    state_history = []
    
    # Reset
    # RslMjEnvWrapper.reset_all() resets buffers AND calls env.reset()
    vec_env.reset_all()
    obs = vec_env.get_observations() # TensorDict
    
    vec_env.env.state
    for step in range(max_steps):
        with torch.no_grad():
            # Get Action
            actions = policy(obs["policy"])
            
            # Step
            obs, rew, done, extras = vec_env.step(actions)
            
        # Store physics state for rendering
        state_history.append(env.state.physics_state.copy())
        
        if step % 50 == 0:
            print(f"Step {step}")

    print("Rendering...")
    
    # Render
    render_data = mujoco.MjData(env.model)
    offsets = get_grid_offsets(num_envs, spacing=grid_spacing)
    frames = []
    
    for i in range(0, len(state_history), decimation):
        print(f"Rendering frame {i}...", end="\r")
        frame = render_many(env.model, render_data, state_history[i], 
                            offsets=offsets, 
                            shape=(640, 480))
        frames.append(frame)
        
    # Save video
    output_path = os.path.join(os.path.dirname(__file__), "play_rollout.mp4")
    print(f"\nSaving video to {output_path}...")
    media.write_video(output_path, frames, fps=video_fps)
    print("Done.")

if __name__ == "__main__":
    main()
