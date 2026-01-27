from typing import Tuple, Dict

import numpy as np
import gymnasium as gym
import mujoco

from dataclasses import dataclass, field
from gs_playground import ROOT_PATH
from gs_playground.src.env import registry
from gs_playground.src.env.base import EnvCfg
from gs_playground.src.env.mujoco_env.mj_env import MjNpEnv, MjNpEnvState

@dataclass
class RewardConfig:
    scales: Dict[str, float] = field(
        default_factory=lambda: {
            "gripper_box": 4.0,
            "box_target": 8.0,
            "no_floor_collision": 5.0,
            "action_rate": -0.0005,
            "lifted_reward": 1.5,
            "success_reward": 20.0,
            "gripper_ctrl": 3.0,
        }
    )
    success_threshold: float = 0.05
    box_init_range: float = 0.05

@dataclass
class ControlConfig:
    # Joint position increment
    action_scale: float = 0.02
    action_history_length: int = 1

@dataclass
class RslPolicyCfg:
    init_noise_std: float = 1.0
    actor_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"
    class_name: str = "ActorCritic"
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True


@dataclass
class RslAlgorithmCfg:
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    class_name: str = "PPO"


@dataclass
class RslRunnerCfg:
    num_steps_per_env: int = 24
    max_iterations: int = 1000
    save_interval: int = 50
    experiment_name: str = "airbot_pick_cube"
    run_name: str = "test_run"
    resume: bool = False
    load_run: int = -1
    checkpoint: int = -1
    resume_path: str = None


@dataclass
class RslTrainCfg:
    policy: RslPolicyCfg = field(default_factory=RslPolicyCfg)
    algorithm: RslAlgorithmCfg = field(default_factory=RslAlgorithmCfg)
    runner: RslRunnerCfg = field(default_factory=RslRunnerCfg)
    obs_groups: dict = field(default_factory=lambda: {"policy": ["policy"]})

@registry.envcfg("airbot_play-pick-cube")
@dataclass
class AirbotPickCubeCfg(EnvCfg):
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    train_cfg: RslTrainCfg = field(default_factory=RslTrainCfg)
    
    model_file: str = (ROOT_PATH / "models" / "robots" / "manipulation" / "airbot_play" / "xmls" / "single_cube.xml").as_posix()
    
    max_episode_seconds: float = 10.0
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02
    
    # Physics parameters
    nconmax: int = 2000
    njmax: int = 500

@registry.env("airbot_play-pick-cube", sim_backend="mujoco")
class AirbotPickCube(MjNpEnv):
    def __init__(self, cfg: AirbotPickCubeCfg, num_envs=1):
        super().__init__(cfg, num_envs)
        
        self.nq = self._model.nq
        self.nv = self._model.nv
        
        self._idx_qpos = 1 
        self._idx_qvel = 1 + self.nq

        self._action_scale = cfg.control_config.action_scale

        self._init_action_space()
        self._init_obs_space()
        
        # Init Keyframe
        key_id = self._model.key("home").id
        self._init_qpos = self._model.key_qpos[key_id].copy()
        
        self._init_sensor_indices()
        self._init_reward_functions()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def update_state(self, state: MjNpEnvState, obs_required: bool = True) -> MjNpEnvState:
        # Cache useful info for logging
        box_pos = state.sensor_data[:, self.idx_box_pos]
        target_pos = state.sensor_data[:, self.idx_target_pos]
        ee_pos = state.sensor_data[:, self.idx_ee_pos]
        
        box_target_dist = np.linalg.norm(box_pos - target_pos, axis=1)
        gripper_box_dist = np.linalg.norm(ee_pos - box_pos, axis=1)
        
        state.info["box_target_dist"] = box_target_dist
        state.info["gripper_box_dist"] = gripper_box_dist
        state.info["is_lifted"] = (box_pos[:, 2] > 0.04).astype(float)
        
        success = box_target_dist < self._cfg.reward_config.success_threshold
        state.info["success"] = success.astype(float)

        # 1. Rewards
        total_reward = np.zeros(self.num_envs, dtype=np.float32)
        log = {}

        actions = getattr(self, "_last_action", None)
        if actions is None:
             actions = np.zeros((self.num_envs, 7))
        
        for name, weight in self._cfg.reward_config.scales.items():
            if name in self._reward_fns:
                r_i = self._reward_fns[name](actions, state)
                total_reward += r_i * weight
                log[f"reward/{name}"] = np.mean(r_i * weight)
        
        # Log extra info
        log["metric/dist_box_target"] = np.mean(box_target_dist)
        log["metric/dist_gripper_box"] = np.mean(gripper_box_dist)
        log["metric/success_rate"] = np.mean(success.astype(float))
        
        state.info["log"] = log
        
        obs = self._compute_obs(state) if obs_required else state.obs
        
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        
        state = state.replace(
            obs=obs,
            reward=total_reward,
            terminated=terminated,
            truncated=truncated
        )
        return state

    def _init_action_space(self):
        # 6 joints + 1 gripper, total 7
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self._num_action = 7

    def _init_obs_space(self):
        # Observation space
        # Joint pos (6) | Gripper pos (1) -> 7
        # Joint vel (6) | Gripper vel (1) -> 7
        # End-effector pos (3) | quat (4) -> 7
        # Box pos (3) | quat (4) -> 7
        # Target pos (3)
        # Relative: Gripper-Box (3), Box-Target (3)
        obs_dim = 7 + 7 + 7 + 7 + 3 + 3 + 3
        
        low = np.full((obs_dim,), -np.inf, dtype=np.float32)
        high = np.full((obs_dim,), np.inf, dtype=np.float32)
        self._observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._num_observation = obs_dim

    def _init_sensor_indices(self):
        super()._init_sensor_indices()
        
        self.idx_joint_pos = [self._get_sensor_slice(f"joint{i}_pos") for i in range(1, 7)]
        self.idx_gripper_pos = self._get_sensor_slice("gripper_pos")
        
        self.idx_joint_vel = [self._get_sensor_slice(f"joint{i}_vel") for i in range(1, 7)]
        self.idx_gripper_vel = self._get_sensor_slice("gripper_vel")
        
        self.idx_ee_pos = self._get_sensor_slice("endpoint_pos")
        self.idx_ee_quat = self._get_sensor_slice("endpoint_quat")
        
        self.idx_box_pos = self._get_sensor_slice("box_pos")
        self.idx_box_quat = self._get_sensor_slice("box_quat")
        
        self.idx_target_pos = self._get_sensor_slice("mocap_target_pos")
        
        self.idx_left_pad_contact = self._get_sensor_slice("left_finger_pad_floor_found")
        self.idx_right_pad_contact = self._get_sensor_slice("right_finger_pad_floor_found")
        
        if "box_zaxis" in self.sensor_indices:
             self.idx_box_zaxis = self._get_sensor_slice("box_zaxis")
        else:
             self.idx_box_zaxis = None

    def _get_sensor_slice(self, name):
        if name not in self.sensor_indices:
            raise ValueError(f"Sensor {name} not found in sensor_indices")
        idx = self.sensor_indices[name]
        adr = self._model.sensor_adr[idx]
        dim = self._model.sensor_dim[idx]
        return slice(adr, adr + dim)

    def _init_reward_functions(self):
        self._reward_fns = {
            "gripper_box": self._reward_gripper_box,
            "box_target": self._reward_box_target,
            "no_floor_collision": self._reward_no_floor_collision,
            "lifted_reward": self._reward_lifted,
            "success_reward": self._reward_success,
            "gripper_ctrl": self._reward_gripper_ctrl,
        }

    def reset(self, env_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        num_resets = len(env_indices)
        if num_resets == 0:
            return None, None, {}

        # 1. Reset Robot to Home
        qpos = np.tile(self._init_qpos, (num_resets, 1))
        qvel = np.zeros((num_resets, self.nv))
        
        # 2. Randomize Box & Target
        box_body_id = self._model.body("box").id
        box_qpos_adr = self._model.jnt_qposadr[self._model.body("box").jntadr[0]]
        init_box_pos = self._init_qpos[box_qpos_adr : box_qpos_adr + 3]
        
        rng = np.random.default_rng()
        box_random = rng.uniform([-0.05, -0.1, 0], [0.05, 0.1, 0], size=(num_resets, 3))
        curr_box_pos = init_box_pos + box_random
        qpos[:, box_qpos_adr : box_qpos_adr + 3] = curr_box_pos
        
        # Target
        target_body = self._model.body("mocap_target")
        target_jnt_adr = target_body.jntadr[0]
        if target_jnt_adr != -1:
             target_qpos_adr = self._model.jnt_qposadr[target_jnt_adr]
             # Random offsets relative to BOX
             target_random = rng.uniform([-0.05, -0.1, 0.1], [0.05, 0.1, 0.2], size=(num_resets, 3))
             qpos[:, target_qpos_adr : target_qpos_adr + 3] = curr_box_pos + target_random

        # Construct new state
        nstate = self.physics_state_dim
        new_states = np.zeros((num_resets, nstate))
        new_states[:, 0] = 0.0
        new_states[:, self._idx_qpos : self._idx_qpos + self.nq] = qpos
        new_states[:, self._idx_qvel : self._idx_qvel + self.nv] = qvel
        
        sensor_batch = np.zeros((num_resets, self._model.nsensordata), dtype=np.float64)
        mj_data = self._worker_data[0]
        for i in range(num_resets):
            mj_data.qpos[:] = qpos[i]
            mj_data.qvel[:] = qvel[i]
            # Use home ctrl
            mj_data.ctrl[:] = self._model.key_ctrl[self._model.key("home").id]
            mujoco.mj_forward(self._model, mj_data)
            sensor_batch[i] = mj_data.sensordata
            
        if hasattr(self, "_state") and self._state is not None:
             self._state.sensor_data[env_indices] = sensor_batch

        obs_state = MjNpEnvState(
             physics_state=new_states, sensor_data=sensor_batch,
             obs=None, reward=None, terminated=None, truncated=None, ctrl=None, info={}
        )
        obs_batch = self._compute_obs(obs_state)
        info = { "success": np.zeros(num_resets) }
        
        return new_states, obs_batch, info

    def apply_action(self, actions: np.ndarray, state: MjNpEnvState) -> np.ndarray:
        # actions: [dq1...dq6, gripper]
        self._last_action = actions
        curr_qpos = []
        for i in range(6):
            curr_qpos.append(state.sensor_data[:, self.idx_joint_pos[i]])
        curr_qpos = np.concatenate(curr_qpos, axis=1) # (B, 6)
        
        d_qpos = actions[:, :6] * self._action_scale
        target_qpos = curr_qpos + d_qpos
        
        # Gripper
        gripper_act = actions[:, 6]
        # Map [-1, 1] to [0, 0.04]
        target_gripper = (gripper_act + 1) / 2 * 0.04
        target_gripper = np.clip(target_gripper, 0, 0.04)
        
        new_ctrl = np.zeros((self._num_envs, 7), dtype=np.float64)
        new_ctrl[:, :6] = target_qpos
        new_ctrl[:, 6] = target_gripper
        
        return new_ctrl

    def _compute_obs(self, state: MjNpEnvState) -> np.ndarray:
        # Joints
        joint_pos = np.concatenate([state.sensor_data[:, self.idx_joint_pos[i]] for i in range(6)], axis=1)
        gripper_pos = state.sensor_data[:, self.idx_gripper_pos]
        
        joint_vel = np.concatenate([state.sensor_data[:, self.idx_joint_vel[i]] for i in range(6)], axis=1)
        gripper_vel = state.sensor_data[:, self.idx_gripper_vel]
        
        ee_pos = state.sensor_data[:, self.idx_ee_pos]
        ee_quat = state.sensor_data[:, self.idx_ee_quat]
        
        box_pos = state.sensor_data[:, self.idx_box_pos]
        box_quat = state.sensor_data[:, self.idx_box_quat]
        
        target_pos = state.sensor_data[:, self.idx_target_pos]
        
        rel_gripper_box = box_pos - ee_pos
        rel_box_target = target_pos - box_pos
        
        obs_list = [
            joint_pos, gripper_pos,
            joint_vel, gripper_vel,
            ee_pos, ee_quat,
            box_pos, box_quat,
            target_pos,
            rel_gripper_box, rel_box_target
        ]
        
        return np.concatenate(obs_list, axis=1)

    def _reward_gripper_box(self, actions, state):
        dist = state.info["gripper_box_dist"]
        return 1.0 - np.tanh(5.0 * dist)
        
    def _reward_box_target(self, actions, state):
        dist = state.info["box_target_dist"]
        return (1.0 - np.tanh(10.0 * dist)) * state.info["is_lifted"]
        
    def _reward_no_floor_collision(self, actions, state):
        left = state.sensor_data[:, self.idx_left_pad_contact].flatten()
        right = state.sensor_data[:, self.idx_right_pad_contact].flatten()
        return -1.0 * ((left > 0) | (right > 0)).astype(float)
        
    def _reward_lifted(self, actions, state):
        box_pos = state.sensor_data[:, self.idx_box_pos]
        target_pos = state.sensor_data[:, self.idx_target_pos]
        z_dist = np.abs(box_pos[:, 2] - target_pos[:, 2])
        return (1.0 - np.tanh(5.0 * z_dist)) * state.info["is_lifted"]
        
    def _reward_success(self, actions, state):
         return state.info["success"]
         
    def _reward_gripper_ctrl(self, actions, state):
         gripper_box_dist = state.info["gripper_box_dist"]
         # gripper ctrl (actuator 6)
         gripper_ctrl = state.ctrl[:, 6]
         # reward if near box, proportional to gripper strength (closing)
         return (gripper_box_dist < 0.02) * (gripper_ctrl > 0.01)
