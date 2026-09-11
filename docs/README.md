# DinoFlow 工作记录

这里记录 `Visuo_Tac` 分支的必要配置、验证结果、训练输出位置和 W&B 项目。

每次训练使用唯一的 `job_name` 和 `outputs/<run-name>/` 目录。训练 stdout/stderr 保存为该目录下的 `train.log`；模型 checkpoint 只写入该目录的 `checkpoints/`，W&B 设置 `disable_artifact=true`，只同步标量指标和配置。

主要记录文件：[training_log.md](training_log.md)。
