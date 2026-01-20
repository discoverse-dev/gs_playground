from gs_playground.src.locomotion.legged_robots.base.legged_robot_cfg import (
    LeggedRobotTorchCfg,
    LeggedRobotCfgPPO,
)
from gs_playground import ROOT_PATH

class G1TrainCfg(LeggedRobotTorchCfg):
    class sim(LeggedRobotTorchCfg.sim):
        dt = 0.005

    class env(LeggedRobotTorchCfg.env):
        num_envs = 10
        num_observations = 47
        space = 1
        num_privileged_obs = 50

    class control(LeggedRobotTorchCfg.control):
        action_scale = 0.25
        decimation = 4
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     } 
        torque_limits = 223.7

    class init_state(LeggedRobotTorchCfg.init_state):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
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
        file = (ROOT_PATH / "models/robots/locomotion/g1/scene.xml").as_posix()
        foot_name = "foot"
        foot_site = ["ll", "rr"]
        body_name = "pelvis"
        # penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["trunk", "knee"]
        ground = "floor"
    
    class rewards(LeggedRobotTorchCfg.rewards):
        class scales(LeggedRobotTorchCfg.rewards.scales):
            hip_pos = -1
            dof_acc = -2.5e-7
            tracking_lin_vel = 1
            contact = 0.18
            feet_air_time = 1
            swing_feet_z = -20
            stand_still = -0.15
            dof_vel = -0.001

    class sensor(LeggedRobotTorchCfg.sensor):
        contact_sensor = True
        local_linvel = "local_linvel"
        gyro = "gyro"
        foot_name = ['fr1', 'fl1']
        num_contact = 1
class G1TrainCfgPPO(LeggedRobotCfgPPO):
        
    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_g1"
        max_iterations = 15000  # number of policy updates