from gs_playground.src.locomotion.legged_robots.base.legged_robot_cfg import (
    LeggedRobotTorchCfg,
    LeggedRobotCfgPPO,
)
from gs_playground import ROOT_PATH

class Go2TrainCfg(LeggedRobotTorchCfg):
    class sim(LeggedRobotTorchCfg.sim):
        dt = 0.005

    class env(LeggedRobotTorchCfg.env):
        num_envs = 100
        num_observations = 49
        space = 1
        num_privileged_obs = 52

    class terrain(LeggedRobotTorchCfg.terrain):
        measure_heights = False
        type = "hfield"
        hfield_path = "/resources/terrain/heightmap_train.hfield"

    class control(LeggedRobotTorchCfg.control):
        action_scale = 0.25
        decimation = 4
        stiffness = 20
        damping = 0.5

    class init_state(LeggedRobotTorchCfg.init_state):
        pos = [-0.0, -0, 0.33]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip": 0.0,  # [rad]
            "RL_hip": 0.0,  # [rad]
            "FR_hip": -0.0,  # [rad]
            "RR_hip": -0.0,  # [rad]
            "FL_thigh": 0.8,  # [rad]
            "RL_thigh": 0.8,  # [rad]
            "FR_thigh": 0.8,  # [rad]
            "RR_thigh": 0.8,  # [rad]
            "FL_calf": -1.8,  # [rad]
            "RL_calf": -1.8,  # [rad]
            "FR_calf": -1.8,  # [rad]
            "RR_calf": -1.8,  # [rad]
        }

    class commands(LeggedRobotTorchCfg.commands):
        class ranges(LeggedRobotTorchCfg.commands.ranges):
            lin_vel_x = [-0, 2]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1, 1]

    class normalization(LeggedRobotTorchCfg.normalization):
        class obs_scales(LeggedRobotTorchCfg.normalization.obs_scales):
            lin_vel = 2
            ang_vel = 0.25
            dof_pos = 1
            dof_vel = 0.05

    class asset(LeggedRobotTorchCfg.asset):
        file = (ROOT_PATH / "models/robots/locomotion/go2/scene_flat.xml").as_posix()
        body_name = 'base'
        foot_name = "foot"
        foot_site = ["FR", "FL", "RR", "RL"]
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "thigh", "calf"]
        ground = "floor"

    class sensor(LeggedRobotTorchCfg.sensor):
        foot_name = ["FR_foot_contact", "FL_foot_contact", "RR_foot_contact", "RL_foot_contact"]
        num_contact = 1

    class rewards(LeggedRobotTorchCfg.rewards):
        class scales(LeggedRobotTorchCfg.rewards.scales):
            hip_pos = -1
            dof_acc = -2.5e-7
            tracking_lin_vel = 1
            contact = 0.18
            feet_air_time = 1
            swing_feet_z = -40
            stand_still = -0.15
            dof_vel = -0.001


class Go2TrainCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_go2"
