from dataclasses import dataclass, field
from gs_playground import ROOT_PATH
from gs_playground.src.env import registry
from gs_playground.src.env.base import EnvCfg

model_file = (ROOT_PATH / "models" / "robots" / "manipulation" / "franka_emika_panda_robotiq" / "xmls" / "single_cube_camera.xml").as_posix()

@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            # Gripper goes to the box.
            "gripper_box": 4.0,
            # Box goes to the target mocap.
            "box_target": 8.0,
            # Do not collide the gripper with the floor.
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
    # Size of cartesian increment.
    action_scale: float = 0.005
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
    experiment_name: str = "franka_pick_rsl"
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


@dataclass
class FrankaCartesianBaseCfg(EnvCfg):
    train_cfg: RslTrainCfg = field(default_factory=RslTrainCfg)
    max_episode_seconds: float = 10.0 # approx 200 steps * 0.05
    model_file: str = model_file
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    
    sim_dt: float = 0.005
    ctrl_dt: float = 0.05
    
    # Physics parameters
    nconmax: int = 12 * 1024
    njmax: int = 128
    
    # Vision
    vision: bool = False
