
import os
import numpy as np
import torch
import gymnasium as gym
from dataclasses import asdict
from tensordict import TensorDict

# Add workspace root to path if needed, though VS Code environment usually handles it.
# Assuming gs_playground is in python path.

from gs_playground.src.locomotion.go2.walk_np import Go2WalkTaskMj
from gs_playground.src.locomotion.go2.cfg import Go2WalkNpEnvCfg
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.env import VecEnv

class RslMjEnvWrapper(VecEnv):
    def __init__(self, env: Go2WalkTaskMj, device: str):
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
        
        # MjNpEnv auto-resets done environments and updates state.obs
        # So we don't need to manually reset here.
        # However, for PPO we might want terminal observations in extras if available.
        # Currently MjNpEnv doesn't seem to provide them explicitly in info, 
        # but for simple walking task it might be fine or we add it later.
        
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

def main():
    # 1. Environment Config
    env_cfg = Go2WalkNpEnvCfg()
    # Adjust config for training if needed
    num_envs = 4096
    
    # 2. Create Environment
    # We instantiate directly to bypass registry string parsing if convenient, 
    # but using registry class is fine.
    env = Go2WalkTaskMj(env_cfg, num_envs=num_envs)
    
    # 3. Wrap Environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vec_env = RslMjEnvWrapper(env, device)
    
    # 4. Train Config
    # Convert dataclass to dict for rsl_rl
    train_cfg = asdict(env_cfg.train_cfg)
    # Flatten runner config into top level for OnPolicyRunner compatibility
    # rsl_rl expects runner params (like num_steps_per_env) at the root of train_cfg
    if "runner" in train_cfg:
        train_cfg.update(train_cfg.pop("runner"))
    
    # 5. Runner
    log_dir = os.path.join(os.path.dirname(__file__), "../../../logs")
    runner = OnPolicyRunner(vec_env, train_cfg, log_dir=log_dir, device=device)
    
    # 6. Learn
    print("Starting training...")
    # Override for testing
    # train_cfg["max_iterations"] = 5
    runner.learn(num_learning_iterations=train_cfg["max_iterations"], init_at_random_ep_len=True)

if __name__ == "__main__":
    main()
