# from legged_gym.envs import *
# from legged_gym.utils import task_registry
from gs_playground.src.env.legged_robots.go1.go1_config import Go1TrainCfg, Go1TrainCfgPPO
from gs_playground.src.env.legged_robots.go1.go1 import Go1_train_env
from datetime import datetime
from gs_playground.addr import GS_GYM_ENVS_DIR
# from legged_gym.rsl_rl.runners import OnPolicyRunner
from gs_playground.src.env.rsl_rl.runners import OnPolicyRunner
import os
import torch
def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result
# def make_alg_runner(
#         env, name=None, train_cfg=None, log_root="default"
#     ) -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:

#         if log_root == "default":
#             log_root = os.path.join(LEGGED_GYM_ENVS_DIR, "logs", train_cfg.runner.experiment_name)
#             log_dir = os.path.join(
#                 log_root,
#                 datetime.now().strftime("%b%d_%H-%M-%S") + "_" + train_cfg.runner.run_name,
#             )
#         elif log_root is None:
#             log_dir = None
#         else:
#             log_dir = os.path.join(
#                 log_root,
#                 datetime.now().strftime("%b%d_%H-%M-%S") + "_" + train_cfg.runner.run_name,
#             )

#         train_cfg_dict = class_to_dict(train_cfg)
#         runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=train_cfg.device)
#         # save resume path before creating a new log_dir
#         resume = train_cfg.runner.resume
#         if resume:
#             # load previously trained model
#             resume_path = get_load_path(
#                 log_root,
#                 load_run=train_cfg.runner.load_run,
#                 checkpoint=train_cfg.runner.checkpoint,
#             )
#             # resume_path = '/home/motphys/train/unitree_rl_gym/deploy/pre_train/g1/motion.pt'
#             print(f"Loading model from: {resume_path}")
#             runner.load(resume_path)
#         return runner, train_cfg
def play(env, cfg, train_cfg):
    # env = task_registry.make_env(name=task, headless=True)
    cfg.env.num_envs = 10
    env = env(Cfg=cfg, headless=False)
    # env.headless = False
    log_dir = None
    # ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=task)
    ppo_runner = OnPolicyRunner(env, class_to_dict(train_cfg), log_dir, device=train_cfg.device)
    # ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations)
    resume_path = "/home/motphys/train/136jdx/gs_playground/src/env/logs/rough_go1/Jan19_13-56-22_/model_1500.pt"
    ppo_runner.load(resume_path)
    obs = env.get_observations()

    policy = ppo_runner.get_inference_policy(env.device)
    print(policy)
    with torch.no_grad():
        while True:
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions)

if __name__ == "__main__":
    # task = "T1"
    play(Go1_train_env, Go1TrainCfg, Go1TrainCfgPPO)

    # /home/motphys/train/136jdx/gs_playground/gs_playground/src/env/resources/robots/go1/scene.xml
