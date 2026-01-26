
import os
import datetime
import numpy as np
import torch
import argparse
import glob
from dataclasses import asdict
from tensordict import TensorDict

from gs_playground.src.env import registry
# Import envs to trigger registration
from gs_playground.src.locomotion.go2.walk_np import Go2WalkTaskMj
from gs_playground.src.locomotion.go1.walk_np import Go1WalkTaskMj
from gs_playground.src.manipulation.tasks.eef.pick_cartesian import FrankaPickCartesian
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.env import VecEnv

class RslMjEnvWrapper(VecEnv):
    def __init__(self, env, device: str):
        self.env = env
        self.device = torch.device(device)

        self.num_envs = env._num_envs
        self.num_actions = env.action_space.shape[0]
        self.num_obs = env.observation_space.shape[0]
        
        print(f"[Wrapper] Physics Engine: CPU (MuJoCo/Numpy)")
        print(f"[Wrapper] RL Training Device: {self.device}")
        
        # Max episode length
        self.max_episode_length = int(env._cfg.max_episode_seconds / env._cfg.ctrl_dt)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Buffers
        self.obs_buf = None
        
        # RSL RL configuration
        self.cfg = {} # Or derive from env cfg
        
        # Initial Reset
        self.reset_all()

    def reset_all(self):
        _, obs, info = self.env.reset(np.arange(self.num_envs))
        self._update_buffers(obs)
        self.episode_length_buf[:] = 0

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # Convert actions to numpy (GPU -> CPU Transfer)
        actions_np = actions.detach().cpu().numpy()
        
        # Step environment (CPU Physics)
        # MjNpEnv returns a state object, not a tuple
        state = self.env.step(actions_np)
        
        obs = state.obs
        reward = state.reward
        terminated = state.terminated
        truncated = state.truncated
        info = state.info
        
        # Handle resets for done environments
        dones = terminated | truncated
        
        # Pass dones to RSL RL (torch) (CPU -> GPU Transfer)
        dones_torch = torch.tensor(dones, device=self.device, dtype=torch.bool)
        rew_torch = torch.tensor(reward, device=self.device, dtype=torch.float)
        
        # Update episode lengths
        self.episode_length_buf += 1
        
        # Reset episode length buffer for done envs
        if np.any(dones):
            reset_indices = np.where(dones)[0]
            # Calculate and log average episode length for environments that just finished
            avg_episode_length = torch.mean(self.episode_length_buf[reset_indices].float())
            self.episode_length_buf[reset_indices] = 0
            
            # Add to logging
            if "log" not in info:
                info["log"] = {}
            info["log"]["train/episode_length"] = avg_episode_length.item()

        # Timeouts
        time_outs = torch.tensor(truncated, device=self.device, dtype=torch.bool)
        
        # Update Torch Buffers (CPU -> GPU Transfer)
        self._update_buffers(obs)
        
        # Extract log from info
        env_log = info.get("log", {})
        
        extras = {
            "time_outs": time_outs,
            "log": env_log
        }
        
        return self.obs_buf, rew_torch, dones_torch, extras

    def get_observations(self) -> TensorDict:
        return self.obs_buf

    def _update_buffers(self, obs: np.ndarray):
        # Efficiently copy numpy array to GPU
        obs_torch = torch.as_tensor(obs, device=self.device, dtype=torch.float)
        # Default group is "policy"
        self.obs_buf = TensorDict(
            {"policy": obs_torch}, 
            batch_size=self.num_envs, 
            device=self.device
        )

def find_checkpoint(root_root, task_name, run_id=None, ckpt_id=None):
    """
    Find checkpoint path.
    root_root: logs/
    task_name: e.g. go2_walk
    run_id: timestamp folder name or None (latest), OR absolute/relative path to run dir or file
    ckpt_id: model_{ckpt_id}.pt or None (latest), OR absolute/relative path to file
    """
    # 1. Direct file check
    if run_id and os.path.isfile(run_id):
        return run_id
    if ckpt_id and os.path.isfile(ckpt_id):
        return ckpt_id

    # 2. Locate Run Directory
    run_dir = None
    
    # If run_id is provided and looks like a directory path
    if run_id and os.path.isdir(run_id):
        run_dir = run_id
    
    # If not resolved, assume standard structure (logs/task_name/run_id)
    if not run_dir:
        task_dir = os.path.join(root_root, task_name)
        if not os.path.exists(task_dir):
            print(f"Task directory {task_dir} not found.")
            return None

        if run_id:
            # run_id is name of folder inside task_dir
            run_dir = os.path.join(task_dir, run_id)
        else:
            # Find latest timestamp
            runs = glob.glob(os.path.join(task_dir, "*"))
            if not runs:
                 print(f"No runs found in {task_dir}")
                 return None
            runs.sort(key=os.path.getmtime, reverse=True)
            run_dir = runs[0]
        
    print(f"Searching in run: {run_dir}")
    
    # Find Model in run_dir
    if ckpt_id:
        if str(ckpt_id).endswith(".pt"):
             model_path = os.path.join(run_dir, ckpt_id)
        else:
             model_path = os.path.join(run_dir, f"model_{ckpt_id}.pt")
    else:
        # Find latest model_*.pt
        models = glob.glob(os.path.join(run_dir, "model_*.pt"))
        if not models:
             print("No models found.")
             return None
             
        def get_iter(f):
            try:
                return int(os.path.basename(f).split('_')[1].split('.')[0])
            except:
                return -1
        models.sort(key=get_iter, reverse=True)
        model_path = models[0]
        
    return model_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task name registered in registry (e.g. go2-flat-terrain-walk, franka-pick-cartesian)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--load_run", type=str, default=None, help="Specific run directory name to resume from (e.g. 2024-01-25_...)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint filename or full path")
    args = parser.parse_args()

    # Adjust config for training if needed
    # num_envs = 2048
    num_envs = 1024
    
    # 2. Create Environment using Registry
    # Ensure imports above have triggered registration
    if not registry.contains(args.task):
         print(f"Error: Task '{args.task}' not found in registry. Available tasks:")
         print(list(registry._envs.keys()))
         return

    env = registry.make(args.task, num_envs=num_envs)
    
    # Access internally stored config
    # MjNpEnv stores it in self._cfg
    env_cfg = env._cfg
    task_name = args.task
    
    # Update experiment name for logging
    if hasattr(env_cfg, "train_cfg") and hasattr(env_cfg.train_cfg, "runner"):
        env_cfg.train_cfg.runner.experiment_name = task_name

    # 3. Wrap Environment
    
    # 3. Wrap Environment
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    vec_env = RslMjEnvWrapper(env, device)
    
    # 4. Train Config
    # Convert dataclass to dict for rsl_rl
    train_cfg = asdict(env_cfg.train_cfg)
    # Flatten runner config into top level for OnPolicyRunner compatibility
    # rsl_rl expects runner params (like num_steps_per_env) at the root of train_cfg
    if "runner" in train_cfg:
        train_cfg.update(train_cfg.pop("runner"))
    
    # 5. Runner
    root_log_dir = os.path.join(os.path.dirname(__file__), "../../../logs")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(root_log_dir, task_name, timestamp)
    
    runner = OnPolicyRunner(vec_env, train_cfg, log_dir=log_dir, device=device)
    
    # 6. Resume Handling
    if args.resume or args.load_run or args.checkpoint:
        print(f"Attempting to resume for task {task_name}...")
        ckpt_path = find_checkpoint(root_log_dir, task_name, args.load_run, args.checkpoint)
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"Loading checkpoint: {ckpt_path}")
            # load_optimizer=True for resuming training
            runner.load(ckpt_path, load_optimizer=True)
            
            # Update log_dir to continue in the SAME directory if resuming specific run
            # Note: OnPolicyRunner creates its own internal logging structures. 
            # If we want to append, we might need more hacky handling or just accept new timestamp folder via 'log_dir'.
            # RSL RL default behavior updates: current_learning_iteration based on loaded checkpoint.
            
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}. Starting from scratch.")

    # 7. Learn
    print("Starting training...")
    # Override for testing
    # train_cfg["max_iterations"] = 5
    runner.learn(num_learning_iterations=train_cfg["max_iterations"], init_at_random_ep_len=True)

if __name__ == "__main__":
    main()
