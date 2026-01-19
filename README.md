###  环境配置

```
uv sync --all-extras --reinstall-package motrixsim
```

### 安装rsl_rl
```
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl && git checkout v1.0.2 && uv pip install -e .
```