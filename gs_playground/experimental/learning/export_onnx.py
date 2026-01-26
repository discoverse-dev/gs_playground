
import os
import sys
import argparse
import torch
import torch.onnx
from dataclasses import asdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from gs_playground.src.locomotion.go2.walk_np import Go2WalkTaskMj
from gs_playground.src.locomotion.go2.cfg import Go2WalkNpEnvCfg
from gs_playground.src.locomotion.go1.walk_np import Go1WalkTaskMj
from gs_playground.src.locomotion.go1.cfg import Go1WalkNpEnvCfg
from gs_playground.experimental.learning.train_rsl_rl import RslMjEnvWrapper
from rsl_rl.runners import OnPolicyRunner

def find_latest_model(log_dir):
    import glob
    files = glob.glob(os.path.join(log_dir, "**", "model_*.pt"), recursive=True)
    if not files: return None
    def extract_key(f):
        base = os.path.basename(f)
        parent = os.path.dirname(f)
        parent_base = os.path.basename(parent)
        try: return (parent_base, int(base.split('_')[1].split('.')[0]))
        except: return (parent_base, -1)
    files.sort(key=extract_key, reverse=True)
    return files[0]

def main():
    parser = argparse.ArgumentParser(description="Export RSL-RL model to ONNX")
    parser.add_argument("--robot", type=str, default="go2", choices=["go1", "go2"], help="Robot type")
    parser.add_argument("--load_model", type=str, default=None, help="Path to the model .pt file to load")
    parser.add_argument("--output_dir", type=str, 
                      default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../sim2sim/onnx")),
                      help="Directory to save the ONNX model")
    args = parser.parse_args()

    if args.robot == "go1":
        env_cfg = Go1WalkNpEnvCfg()
        EnvClass = Go1WalkTaskMj
        log_folder = "go1_walk"
        out_filename = "go1_policy.onnx"
    else:
        env_cfg = Go2WalkNpEnvCfg()
        EnvClass = Go2WalkTaskMj
        log_folder = "go2_walk"
        out_filename = "go2_policy.onnx"

    if args.load_model:
        model_path = args.load_model
    else:
        # Default fallback
        logs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../../logs/{log_folder}"))
        if os.path.exists(logs_root):
             model_path = find_latest_model(logs_root)
        else:
             model_path = None

    if not model_path or not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please provide a valid model path using --load_model")
        return

    print(f"Loading {model_path}")
    
    # Mock num_envs needed for wrapper but we don't simulate
    env = EnvClass(env_cfg, num_envs=1)
    
    # Exporting for CPU usually safer/more portable for onnxruntime cpu provider
    device = "cpu" 
    vec_env = RslMjEnvWrapper(env, device)
    
    train_cfg = asdict(env_cfg.train_cfg)
    if "runner" in train_cfg: train_cfg.update(train_cfg.pop("runner"))
    
    # We need a dummy log_dir for runner init
    runner_log_dir = os.path.dirname(os.path.dirname(model_path))
    runner = OnPolicyRunner(vec_env, train_cfg, log_dir=runner_log_dir, device=device)
    loaded_dict = runner.load(model_path, load_optimizer=False)
    
    # Correct attribute path based on previous debug
    if hasattr(runner.alg, "policy"):
        actor_critic = runner.alg.policy
    else:
        # Some versions might store it differently
        actor_critic = runner.alg.actor_critic
        
    class OnnxPolicy(torch.nn.Module):
        def __init__(self, actor_critic):
            super().__init__()
            self.actor = actor_critic.actor
            # Handle RSL-RL empirical normalization
            self.normalizer = actor_critic.actor_obs_normalizer # Typically an EmpiricalNormalization module or Identity
            
        def forward(self, obs):
            # 1. Normalize
            obs_norm = self.normalizer(obs)
            # 2. Forward actor
            return self.actor(obs_norm)
    
    # Ensure normalizer is in eval mode (using stats from checkpoint)
    if hasattr(actor_critic, "actor_obs_normalizer"):
        actor_critic.actor_obs_normalizer.eval()

    policy_export = OnnxPolicy(actor_critic)
    policy_export.eval()
    policy_export.to(device)
    
    # Input shape: 48 (based on observation space)
    # 3 (linvel) + 3 (gyro) + 3 (grav) + 12 (pos) + 12 (vel) + 12 (action) + 3 (cmd) = 48
    dummy_input = torch.randn(1, 48, device=device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, out_filename)
    
    print(f"Exporting to {output_path}")
    torch.onnx.export(
        policy_export, 
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
