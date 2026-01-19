from gs_playground.src.locomotion.legged_robots.go1.go1_config import Go1TrainCfg, Go1TrainCfgPPO
from gs_playground.src.locomotion.legged_robots.go1.go1 import Go1_train_env
from rsl_rl.runners import OnPolicyRunner

import torch
import argparse

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

def play(env, cfg, train_cfg, resume_path, num_envs):
    # env = task_registry.make_env(name=task, headless=True)
    cfg.env.num_envs = num_envs
    env = env(Cfg=cfg, headless=False)
    # env.headless = False
    log_dir = None
    ppo_runner = OnPolicyRunner(env, class_to_dict(train_cfg), log_dir, device=train_cfg.device)
    ppo_runner.load(resume_path)
    obs = env.get_observations()

    policy = ppo_runner.get_inference_policy(env.device)
    print(policy)
    with torch.no_grad():
        while True:
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_path', type=str, required=True, help='Path to the model to resume from')
    parser.add_argument('--num_envs', type=int, default=10, help='Number of environments')
    args = parser.parse_args()

    play(Go1_train_env, Go1TrainCfg, Go1TrainCfgPPO, args.resume_path, args.num_envs)
