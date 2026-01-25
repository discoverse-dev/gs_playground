
import os
import sys
import torch
import torch.onnx
from dataclasses import asdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from gs_playground.src.locomotion.go2.walk_np import Go2WalkTaskMj
from gs_playground.src.locomotion.go2.cfg import Go2WalkNpEnvCfg
from gs_playground.experimental.learning.train_rsl_rl import RslMjEnvWrapper
from rsl_rl.runners import OnPolicyRunner

def find_latest_model(log_dir):
    import glob
    files = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if not files: return None
    def extract_iter(f):
        base = os.path.basename(f)
        try: return int(base.split('_')[1].split('.')[0])
        except: return -1
    files.sort(key=extract_iter, reverse=True)
    return files[0]

def main():
    logs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../logs"))
    model_path = find_latest_model(logs_root)
    if not model_path:
        print("Model not found")
        return

    print(f"Loading {model_path}")
    
    env_cfg = Go2WalkNpEnvCfg()
    # Mock num_envs needed for wrapper but we don't simulate
    env = Go2WalkTaskMj(env_cfg, num_envs=1)
    
    # Exporting for CPU usually safer/more portable for onnxruntime cpu provider
    device = "cpu" 
    vec_env = RslMjEnvWrapper(env, device)
    
    train_cfg = asdict(env_cfg.train_cfg)
    if "runner" in train_cfg: train_cfg.update(train_cfg.pop("runner"))
    
    runner = OnPolicyRunner(vec_env, train_cfg, log_dir=logs_root, device=device)
    runner.load(model_path, load_optimizer=False)
    
    # Correct attribute path based on previous debug
    if hasattr(runner.alg, "policy"):
        policy = runner.alg.policy.actor
    else:
        policy = runner.alg.actor_critic.actor
        
    policy.eval()
    policy.to(device)
    
    # Input shape: 48 (based on observation space)
    # 3 (linvel) + 3 (gyro) + 3 (grav) + 12 (pos) + 12 (vel) + 12 (action) + 3 (cmd) = 48
    dummy_input = torch.randn(1, 48, device=device)
    
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../sim2sim/motrix/go2_policy.onnx"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Exporting to {output_path}")
    torch.onnx.export(
        policy, 
        dummy_input, 
        output_path, 
        verbose=False,
        input_names=['obs'], 
        output_names=['actions'],
        opset_version=11
    )
    print("Export complete.")

if __name__ == "__main__":
    main()
