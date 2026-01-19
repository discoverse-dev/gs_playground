from gs_playground.src.locomotion.legged_robots.go1.go1_config import Go1TrainCfg, Go1TrainCfgPPO
from gs_playground.src.locomotion.legged_robots.go1.go1 import Go1_train_env
from datetime import datetime
from gs_playground.addr import GS_GYM_ENVS_DIR
from rsl_rl.runners import OnPolicyRunner

import os
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
   
    cfg.env.num_envs = 1024
    env = env(Cfg=cfg, headless=headless)
 
    log_root = os.path.join(GS_GYM_ENVS_DIR, "logs", train_cfg.runner.experiment_name)
    log_dir = os.path.join(
        log_root,
        datetime.now().strftime("%b%d_%H-%M-%S") + "_" + train_cfg.runner.run_name,
    )
    ppo_runner = OnPolicyRunner(env, class_to_dict(train_cfg), log_dir, device=train_cfg.device)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations)

if __name__ == "__main__":
    train(Go1_train_env, Go1TrainCfg, Go1TrainCfgPPO, True)
