
import os
import sys
import argparse
import numpy as np
import torch
import mujoco
from dataclasses import asdict
import mediapy as media
import math
import glob
import multiprocessing
from functools import partial

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

# Worker global context
_worker_ctx = {}

def init_worker(model_path, shape):
    """Initialize MuJoCo context for worker process."""
    _worker_ctx['model'] = mujoco.MjModel.from_xml_path(model_path)
    _worker_ctx['data'] = mujoco.MjData(_worker_ctx['model'])
    _worker_ctx['renderer'] = mujoco.Renderer(_worker_ctx['model'], height=shape[1], width=shape[0])

def render_frame_job(args):
    """
    Worker function to render a single frame.
    args: (state_batch, offsets, transparent, shape)
    """
    state_batch, offsets, transparent = args
    
    model = _worker_ctx['model']
    data = _worker_ctx['data']
    renderer = _worker_ctx['renderer']
    
    # Visual options
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
    pert = mujoco.MjvPerturb()
    catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

    # Helper to set state
    def set_state(d, s, offset=None):
        d.time = s[0]
        d.qpos[:] = s[1:1+model.nq]
        d.qvel[:] = s[1+model.nq:1+model.nq+model.nv]
        if offset is not None:
            d.qpos[0] += offset[0]
            d.qpos[1] += offset[1]
        mujoco.mj_forward(model, d)
        
    num_envs = state_batch.shape[0]

    # 1. Clear/Init Scene
    set_state(data, state_batch[0], offsets[0] if offsets is not None else None)
    
    # Init Camera
    cam = mujoco.MjvCamera()
    if offsets is not None:
        center_x = np.mean(offsets[:, 0])
        center_y = np.mean(offsets[:, 1])
        cam.lookat = [center_x, center_y, 0.0]
        cam.distance = 4.5
        cam.elevation = -15
        cam.azimuth = 90
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    renderer.update_scene(data, camera=cam, scene_option=vopt)
    
    # 2. Add other robots
    for i in range(1, num_envs):
        set_state(data, state_batch[i], offsets[i] if offsets is not None else None)
        mujoco.mjv_addGeoms(model, data, vopt, pert, catmask, renderer.scene)

    return renderer.render()

def render_many(model, data, state_batch, shape=(640, 480), transparent=False, offsets=None):
    """
    Deprecated: Serial render function. kept for reference or fallback.
    """
    pass

def find_latest_model(log_dir):
    # Search for model_*.pt recursively in log_dir
    # Structure: log_dir / experiment_name / timestamp / model_*.pt
    search_pattern = os.path.join(log_dir, "**", "model_*.pt")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        return None
    
    # Sort by (timestamp_dir, iteration)
    def extract_key(f):
        # Parent dirname is usually timestamp-like "2026-01-25_..."
        parent = os.path.dirname(f)
        parent_name = os.path.basename(parent)
        
        # Iteration from filename "model_100.pt"
        base = os.path.basename(f)
        try:
            iteration = int(base.split('_')[1].split('.')[0])
        except:
            iteration = -1
        
        # We sort by parent dir name (timestamp) then iteration
        return (parent_name, iteration)
            
    files.sort(key=extract_key, reverse=True)
    print(f"Found latest model: {files[0]}")
    return files[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to model checkoint")
    args = parser.parse_args()

    # Config
    num_envs = 4 # Visualizing 4 dogs
    max_steps = 300 # 300 steps
    grid_spacing = 1.0
    video_fps = 25
    decimation = 2 # Render every 2nd step (50Hz control -> 25fps)
    
    # Paths
    logs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../logs"))
    
    if args.model:
        model_path = args.model
    else:
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
            actions[:] = 0.0
            # Step
            obs, rew, done, extras = vec_env.step(actions)
            
        # Store physics state for rendering
        state_history.append(env.state.physics_state.copy())
        
        if step % 50 == 0:
            print(f"Step {step}")

    print("Rendering...")
    
    offsets = get_grid_offsets(num_envs, spacing=grid_spacing)
    
    # Prepare arguments for multiprocessing
    # Filter state history by decimation
    render_states = state_history[::decimation]
    render_shape = (640, 480)
    
    # args: (state_batch, offsets, transparent)
    pool_args = [(s, offsets, False) for s in render_states]
    
    # Use multiprocessing Pool to render frames in parallel
    num_workers = min(multiprocessing.cpu_count(), 8) # Cap workers
    print(f"Starting render pool with {num_workers} workers...")
    
    with multiprocessing.Pool(processes=num_workers, 
                              initializer=init_worker, 
                              initargs=(env_cfg.model_file, render_shape)) as pool:
        
        frames = pool.map(render_frame_job, pool_args)
        
    print(f"Rendered {len(frames)} frames.")
        
    # Save video
    output_path = os.path.join(os.path.dirname(__file__), "play_rollout.mp4")
    print(f"\nSaving video to {output_path}...")
    media.write_video(output_path, frames, fps=video_fps)
    print("Done.")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
