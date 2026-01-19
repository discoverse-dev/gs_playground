from gs_playground import ROOT_PATH
from gs_playground.src.locomotion.legged_robots.base.legged_robot_cfg import (
    LeggedRobotTorchCfg,
    LeggedRobotCfgPPO,
)

class Go2TrainCfg(LeggedRobotTorchCfg):
    class sim(LeggedRobotTorchCfg.sim):
        dt = 0.005

    class env(LeggedRobotTorchCfg.env):
        num_envs = 100
        num_observations = 49
        space = 1
        num_privileged_obs = 52

    class control(LeggedRobotTorchCfg.control):
        action_scale = 0.25
        decimation = 4
        stiffness = 20
        damping = 0.5

    class init_state(LeggedRobotTorchCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            "FL_hip"    :  0.1,  # [rad]
            "FL_thigh"  :  0.9,  # [rad]
            "FL_calf"   : -1.8,  # [rad]
            "FR_hip"    : -0.1,  # [rad]
            "FR_thigh"  :  0.9,  # [rad]
            "FR_calf"   : -1.8,  # [rad]
            "RL_hip"    :  0.1,  # [rad]
            "RL_thigh"  :  0.9,  # [rad]
            "RL_calf"   : -1.8,  # [rad]
            "RR_hip"    : -0.1,  # [rad]
            "RR_thigh"  :  0.9,  # [rad]
            "RR_calf"   : -1.8,  # [rad]
        }

    class commands(LeggedRobotTorchCfg.commands):
        class ranges(LeggedRobotTorchCfg.commands.ranges):
            lin_vel_x   = [-2, 2]
            lin_vel_y   = [-1, 1]
            ang_vel_yaw = [-3.1416, 3.1416]

    class normalization(LeggedRobotTorchCfg.normalization):
        class obs_scales(LeggedRobotTorchCfg.normalization.obs_scales):
            lin_vel = 2
            ang_vel = 0.25
            dof_pos = 1
            dof_vel = 0.05

    class asset(LeggedRobotTorchCfg.asset):
        file = (ROOT_PATH / "models/robots/locomotion/go2/scene_flat.xml").as_posix()
        body_name = "base"
        foot_name = "foot"
        foot_site = ["FR", "FL", "RR", "RL"]
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = [
            "base_collision_0", "base_collision_1", "base_collision_2",
            "fl_hip_0", "fr_hip_0", "rl_hip_0", "rr_hip_0",
        ]
        ground = "floor"

    class sensor:
        local_linvel = "local_linvel"
        gyro = "gyro"
        contact_sensor = True
        foot_name = [
            "FL_foot_contact", 
            "FR_foot_contact", 
            "RL_foot_contact", 
            "RR_foot_contact"
        ]
        num_contact = 1

    class rewards(LeggedRobotTorchCfg.rewards):
        class scales(LeggedRobotTorchCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            stand_still = -0.
            dof_vel = -0.
            dof_acc = -2.5e-7
            hip_pos = -1

            feet_air_time = 1.
            swing_feet_z = -40
    
    class lidarcfg:
        lidartype: str = "mid360"
        downsample: int = 1
        dynamic_lidar: bool = False

class Go2TrainCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_go2"
