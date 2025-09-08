

# DeepSpeed Day2 学习笔记

## 修改训练循环

optimizer.zero_grad() → 自动处理
loss.backward() → model_engine.backward(loss)
optimizer.step() → model_engine.step()

## 设备管理

.to(device) → .to(model_engine.local_rank)

## 配置文件要点

train_batch_size: 全局批次大小
train_micro_batch_size_per_gpu: 每个GPU的微批次
gradient_accumulation_steps: 梯度累积步数
zero_optimization.stage: Zero优化级别

## 启动
```bash
deepspeed --master_port=29501 deepspeed_training.py
```

