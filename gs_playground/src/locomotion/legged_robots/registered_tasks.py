from gs_playground.src.locomotion.task_registry import task_registry

# Register Go1
from gs_playground.src.locomotion.legged_robots.go1.go1 import Go1_train_env
from gs_playground.src.locomotion.legged_robots.go1.go1_config import Go1TrainCfg, Go1TrainCfgPPO

task_registry.register("go1", Go1_train_env, Go1TrainCfg, Go1TrainCfgPPO)

# Register Go2
from gs_playground.src.locomotion.legged_robots.go2.go2_flat import Go2_train_env
from gs_playground.src.locomotion.legged_robots.go2.go2_config import Go2TrainCfg, Go2TrainCfgPPO

task_registry.register("go2_flat", Go2_train_env, Go2TrainCfg, Go2TrainCfgPPO)
    
from gs_playground.src.locomotion.legged_robots.go2.go2_lidar import Go2_lidar_train_env
task_registry.register("go2_lidar", Go2_lidar_train_env, Go2TrainCfg, Go2TrainCfgPPO)

# Register G1
