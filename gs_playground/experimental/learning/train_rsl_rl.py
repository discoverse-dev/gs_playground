
import os
import datetime
import numpy as np
import torch
import argparse
import glob
from dataclasses import asdict
from tensordict import TensorDict

from gs_playground.src.locomotion.go2.walk_np import Go2WalkTaskMj
from gs_playground.src.locomotion.go2.cfg import Go2WalkNpEnvCfg
from gs_playground.src.locomotion.go1.walk_np import Go1WalkTaskMj
from gs_playground.src.locomotion.go1.cfg import Go1WalkNpEnvCfg
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
    run_id: timestamp folder name or None (latest)
    ckpt_id: model_{ckpt_id}.pt or None (latest)
    """
    search_dir = root_root
    task_dirs = glob.glob(os.path.join(root_root, "*")) # e.g. logs/go2_walk
    if not task_dirs:
        print(f"No task directories found in {root_root}")
        return None
        
    task_dir = os.path.join(root_root, task_name)
    if not os.path.exists(task_dir):
        # If specific task dir doesn't exist, we can't resume for that task
        print(f"Task directory {task_dir} not found.")
        return None

    # Find Run Dir
    if run_id:
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
    
    # Find Model
    if ckpt_id:
        # Check specific
        if str(ckpt_id).endswith(".pt"):
             model_path = ckpt_id if os.path.isabs(ckpt_id) else os.path.join(run_dir, ckpt_id)
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
    parser.add_argument("--robot", type=str, default="go2", choices=["go1", "go2"], help="Robot to train: go1 or go2")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--load_run", type=str, default=None, help="Specific run directory name to resume from (e.g. 2024-01-25_...)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint filename or full path")
    args = parser.parse_args()

    # 1. Environment Config
    if args.robot == "go1":
        env_cfg = Go1WalkNpEnvCfg()
        EnvClass = Go1WalkTaskMj
        task_name = "go1_walk"
        env_cfg.train_cfg.runner.experiment_name = task_name
    elif args.robot == "go2":
        env_cfg = Go2WalkNpEnvCfg()
        EnvClass = Go2WalkTaskMj
        task_name = "go2_walk"
        env_cfg.train_cfg.runner.experiment_name = task_name
    else:
        raise ValueError(f"Unknown robot: {args.robot}")

    # Adjust config for training if needed
    num_envs = 2048
    
    # 2. Create Environment
    # We instantiate directly to bypass registry string parsing if convenient, 
    # but using registry class is fine.
    env = EnvClass(env_cfg, num_envs=num_envs)
    
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
