from gs_playground import ROOT_PATH
from datetime import datetime
from rsl_rl.runners import OnPolicyRunner
from gs_playground.src.locomotion.task_registry import task_registry
import gs_playground.src.locomotion.legged_robots.registered_tasks

import os
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

def train(env, cfg, train_cfg, headless):
   
    env = env(Cfg=cfg, headless=headless)
 
    log_root = os.path.join((ROOT_PATH / "../logs").as_posix(), train_cfg.runner.experiment_name)
    log_dir = os.path.join(
        log_root,
        datetime.now().strftime("%b%d_%H-%M-%S") + "_" + train_cfg.runner.run_name,
    )
    ppo_runner = OnPolicyRunner(env, class_to_dict(train_cfg), log_dir, device=train_cfg.device)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='go1', help='Task name')
    parser.add_argument('--run_name', type=str, default='', help='Run name')
    parser.add_argument('--num_envs', type=int, default=1024, help='num envs')
    parser.add_argument('--headless', type=bool, default=False, help='headless')
    args = parser.parse_args()

    env_cls, env_cfg, train_cfg = task_registry.get_task(args.task)
    if args.run_name:
        train_cfg.runner.run_name = args.run_name
    env_cfg.env.num_envs = args.num_envs
        
    train(env_cls, env_cfg, train_cfg, args.headless)
