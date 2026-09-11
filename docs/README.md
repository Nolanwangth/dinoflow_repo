# DinoFlow baseline 实验记录

这里记录 `visuo-basline` 分支的代码变更、batch 探测、训练启动参数、训练输出位置和验证结果。

每次训练使用唯一的 `job_name` 和 `outputs/<run-name>/` 目录。训练 stdout/stderr 保存为该目录下的 `train.log`；模型 checkpoint 只写入该目录的 `checkpoints/`，W&B 设置 `disable_artifact=true`，只同步标量指标和配置。

主要记录文件：[training_log.md](training_log.md)。
