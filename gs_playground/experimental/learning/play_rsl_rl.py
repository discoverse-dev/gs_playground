
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

from gs_playground.src.env import registry
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
        
        apply_root_offset = False
        
        if offset is not None:
            # Check if Root (Body 1) has a free joint or slide joints allowing X/Y movement
            # Body 0 is world. Body 1 is usually the robot base.
            robot_moved = False
            
            # Heuristic: Check joint at qpos 0, 1. 
            # If jnt_type[0] is free (0), fine.
            # If jnt_type[0] is slide (2) and axis is x/y...
            
            # Better check: Does the first body have a joint?
            first_body_jnt = model.body_jntadr[1] if model.nbody > 1 else -1
            if first_body_jnt >= 0:
                jnt_type = model.jnt_type[first_body_jnt]
                # mjJNT_FREE=0
                if jnt_type == 0:
                     d.qpos[0] += offset[0]
                     d.qpos[1] += offset[1]
                     robot_moved = True
            
            # If robot wasn't moved via qpos, we need to manually offset geometries later
            if not robot_moved:
                apply_root_offset = True

            # 2. Box offset
            box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
            if box_id >= 0:
                jnt_adr = model.body_jntadr[box_id]
                if jnt_adr >= 0:
                    qpos_adr = model.jnt_qposadr[jnt_adr]
                    d.qpos[qpos_adr] += offset[0]
                    d.qpos[qpos_adr+1] += offset[1]

            # 3. Target offset (target_x, target_y)
            target_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_x")
            if target_x >= 0:
                 d.qpos[model.jnt_qposadr[target_x]] += offset[0]
            
            target_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_y")
            if target_y >= 0:
                 d.qpos[model.jnt_qposadr[target_y]] += offset[1]

        mujoco.mj_forward(model, d)
        
        # Post-process: Shift all geometries if robot root wasn't moved
        if apply_root_offset and offset is not None:
             # Shift all geoms? 
             # We should shift Everything that is PART OF THE ROBOT.
             # Or just everything? 
             # Box and Target were already shifted via qpos. 
             # BUT qpos shift updates body_pos which updates geom_pos.
             # If we shift ALL geom_pos, we double shift Box and Target!
             
             # So we need to shift geoms that belong to bodies which are NOT Box or Target.
             # Or simpler: Shift everything, but subtract offset from Box/Target qpos first? No.
             
             # Let's iterate bodies.
             # Simple heuristic: Shift everything except Box and Target?
             box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
             target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mocap_target")
             
             # Also target might be just a body named "mocap_target"
             
             for i in range(model.ngeom):
                 body_id = model.geom_bodyid[i]
                 # If it is robot body. 
                 # We want to shift generally everything that wasn't shifted by Qpos.
                 # Box and Target were shifted by Qpos.
                 # Floor (Plane) should usually NOT be shifted (infinite).
                 # Everything else (Robot Base, Robot Links, Decoration) should be shifted.
                 
                 is_box_or_target = (body_id == box_body_id) or (body_id == target_body_id)
                 is_plane = (model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE)
                 
                 if not is_box_or_target and not is_plane:
                      d.geom_xpos[i, 0] += offset[0]
                      d.geom_xpos[i, 1] += offset[1]
             
             # Also update site positions if they are visualized
             for i in range(model.nsite):
                 body_id = model.site_bodyid[i]
                 
                 is_box_or_target = (body_id == box_body_id) or (body_id == target_body_id)
                 
                 if not is_box_or_target:
                      d.site_xpos[i, 0] += offset[0]
                      d.site_xpos[i, 1] += offset[1]
        
    num_envs = state_batch.shape[0]

    # 1. Clear/Init Scene
    set_state(data, state_batch[0], offsets[0] if offsets is not None else None)
    
    # Init Camera
    cam = mujoco.MjvCamera()
    if offsets is not None:
        center_x = np.mean(offsets[:, 0])
        center_y = np.mean(offsets[:, 1])
        cam.lookat = [center_x, center_y, 0.0]
        # cam.lookat = [0.65, 0, 0.2]
        cam.distance = 3.0
        cam.elevation = -20
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
    parser.add_argument("--task", type=str, required=True, help="Task name registered in registry")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkoint")
    args = parser.parse_args()

    # Config
    num_envs = 4 #64
    max_steps = 300 # 300 steps
    grid_spacing = 1.0 / 2.
    video_fps = 25
    decimation = 2 # Render every 2nd step (50Hz control -> 25fps)
    
    # # Check if task is fixed base
    # is_fixed_base = "airbot" in args.task or "franka" in args.task # Simple heuristic or check model
    # if is_fixed_base:
    #     print("Detected fixed base robot. Disabling grid offset.")
    #     grid_spacing = 0.0
        
    # Paths
    logs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../logs"))
    
    if args.model:
        model_path = args.model
    else:
        # Update search logic to filter by robot experiment name if possible, 
        # or just find latest. Ideally should filter by robot
        model_path = find_latest_model(logs_root)
    
    if not model_path:
        print(f"No model found in {logs_root}")
        return
        
    print(f"Loading model: {model_path}")
    
    # 1. Environment using Registry
    if not registry.contains(args.task):
         print(f"Error: Task '{args.task}' not found in registry. Available tasks:")
         print(list(registry._envs.keys()))
         return

    print(f"Initializing Env ({args.task}) with {num_envs} envs...")
    env = registry.make(args.task, num_envs=num_envs)
    env_cfg = env._cfg # Get config from instance
    
    # Force zero commands for playback if applicable
    if hasattr(env_cfg, "commands"):
        # env_cfg.commands.vel_limit is a list [[min], [max]]
        # We set min and max to 0 to ensure sampled commands are always 0
        if hasattr(env_cfg.commands, "vel_limit"):
             env_cfg.commands.vel_limit = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

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
        
        # Load empirical normalization stats if available
        if 'model_state_dict' in loaded_dict:
             state_dict = loaded_dict['model_state_dict']
             # RSL-RL empirical normalization saves running mean and var in the model state dict
             # The keys would be 'std.running_mean_var.running_mean' etc if it was a standalone module,
             # but inside ActorCritic it's likely under 'actor_obs_normalizer'
             pass
    else:
        print("Model loaded (no dict returned).")
    
    policy = runner.alg.policy
    policy.eval()

    # Need to move normalizer to eval mode to stop updating stats
    if hasattr(policy, 'actor_obs_normalizer'):
         policy.actor_obs_normalizer.eval()
         print("Actor observation normalizer set to eval mode.")
    
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
            # Use act_inference to ensure observation normalization is applied if configured
            actions = policy.act_inference(obs)
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
