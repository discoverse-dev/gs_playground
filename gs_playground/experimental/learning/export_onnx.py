
import os
import sys
import argparse
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
    parser = argparse.ArgumentParser(description="Export RSL-RL model to ONNX")
    parser.add_argument("--load_model", type=str, default=None, help="Path to the model .pt file to load")
    parser.add_argument("--output_dir", type=str, 
                      default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../sim2sim/onnx")),
                      help="Directory to save the ONNX model")
    args = parser.parse_args()

    if args.load_model:
        model_path = args.load_model
    else:
        # Default fallback: try to find latest in logs/go2_walk
        logs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../logs/go2_walk"))
        # We need to search recursively or just pick the latest timestamp folder
        # Simplified: just look in logs_root if user didn't structure it, or ask user to provide path if complex.
        # Given the new structure logs/go2_walk/<timestamp>/model_*.pt
        # Let's try to find the latest timestamp folder first
        if os.path.exists(logs_root):
             timestamps = sorted([d for d in os.listdir(logs_root) if os.path.isdir(os.path.join(logs_root, d))])
             if timestamps:
                 latest_log_dir = os.path.join(logs_root, timestamps[-1])
                 model_path = find_latest_model(latest_log_dir)
             else:
                 model_path = None
        else:
             model_path = None

    if not model_path or not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please provide a valid model path using --load_model")
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
    
    # We need a dummy log_dir for runner init
    runner_log_dir = os.path.dirname(os.path.dirname(model_path))
    runner = OnPolicyRunner(vec_env, train_cfg, log_dir=runner_log_dir, device=device)
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
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "go2_policy.onnx")
    
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
