from gs_playground.src.locomotion.legged_robots.go1.go1_config import Go1TrainCfg, Go1TrainCfgPPO
from gs_playground.src.locomotion.legged_robots.go1.go1 import Go1_train_env
from datetime import datetime
from gs_playground.addr import GS_GYM_ENVS_DIR
from rsl_rl.runners import OnPolicyRunner

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

def play(env, cfg, train_cfg):
    # env = task_registry.make_env(name=task, headless=True)
    cfg.env.num_envs = 10
    env = env(Cfg=cfg, headless=False)
    # env.headless = False
    log_dir = None
    ppo_runner = OnPolicyRunner(env, class_to_dict(train_cfg), log_dir, device=train_cfg.device)
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
    play(Go1_train_env, Go1TrainCfg, Go1TrainCfgPPO)
