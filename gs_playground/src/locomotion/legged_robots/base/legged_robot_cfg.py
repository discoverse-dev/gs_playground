from gs_playground import ROOT_PATH

class LeggedRobotTorchCfg:
    class sim:
        max_episode_length = 1000
        max_episode_length_s = 1000
        dt = 0.004
        device = "cpu"

    class env:
        num_envs = 100
        num_observations = 48
        num_actions = 12
        space = 1
        num_privileged_obs = None

    class init_state:
        # the initial position of the robot in the world frame
        pos = [0.0, 0.0, 33]

        # the default angles for all joints. key = joint name, value = target angle [rad]
        default_joint_angles = {
            "FL_hip": 0.1,
            "RL_hip": 0.1,
            "FR_hip": -0.1,
            "RR_hip": -0.1,
            "FL_thigh": 0.8,
            "RL_thigh": 1.0,
            "FR_thigh": 0.8,
            "RR_thigh": 1.0,
            "FL_calf": -1.5,
            "RL_calf": -1.5,
            "FR_calf": -1.5,
            "RR_calf": -1.5,
        }

    class terrain:
        measure_heights = False
        type = "plane"

    class commands:
        resampling_interval = 200  # dt

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class control:
        # PD Drive parameters:
        control_type = "P"
        stiffness = 35  # [N*m/rad]
        damping = 0  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10
        torque_limits = 23.7
        clip_actions = 100.0

    class asset:
        file = (ROOT_PATH / "models/robots/locomotion/go1/scene.xml").as_posix()
        body_name = "trunk"
        # foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = [
            "trunk",
        ]
        ground = "floor"

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 1.0

    class domain_rand:
        # randomize_friction = True
        # friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
            
    class sensor:
        local_linvel = "local_linvel"
        gyro = "gyro"
        contact_sensor = True
        foot_name = ["fr1", "fl1", "rr1", "rl1"]
        num_contact = 1

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.0
            torques = -0.00001
            dof_vel = -0.0
            dof_acc = -2.5e-7
            base_height = -0.0
            feet_air_time = 1.0
            collision = -1.0 * 0
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.0
            hip_pos = -1
            calf_pos = -0.3 * 0

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = (
            1.0  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 1.0
        max_contact_force = 100.0  # forces above this value are penalized


class LeggedRobotCfgPPO:
    seed = 1
    runner_class_name = "OnPolicyRunner"
    device = "cuda"

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24  # per iteration
        max_iterations = 1500  # number of policy updates

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = "test"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
