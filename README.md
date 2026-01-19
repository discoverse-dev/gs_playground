###  环境配置

```bash
uv sync --all-extras --reinstall-package motrixsim
```

### 安装rsl_rl
```bash
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl && git checkout v1.0.2 && uv pip install -e .
```

```bash
cd gs_playground/src/locomotion/scripts
python train.py
```

```bash
python gs_playground/src/locomotion/scripts/play.py \
    --resume_path "/path/to/your/model.pt" \
    --num_envs 10
```