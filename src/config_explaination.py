








"""
DeepSpeed配置参数详细解释
"""

def explain_config_sections():
    explanations = {
        "batch_settings": {
            "train_batch_size": "全局训练批次大小（所有GPU总和）",
            "train_micro_batch_size_per_gpu": "每个GPU的微批次大小",
            "gradient_accumulation_steps": "梯度累积步数，train_batch_size = micro_batch * num_gpus * gradient_accumulation_steps"
        },
        
        "optimizer_settings": {
            "type": "优化器类型：Adam, AdamW, SGD, Lamb等",
            "lr": "学习率",
            "betas": "Adam的beta参数 [beta1, beta2]",
            "eps": "数值稳定性参数",
            "weight_decay": "权重衰减（L2正则化）"
        },
        
        "scheduler_settings": {
            "WarmupLR": "线性预热学习率",
            "WarmupDecayLR": "预热后线性衰减",
            "WarmupCosineLR": "预热后余弦退火",
            "warmup_min_lr": "预热开始学习率",
            "warmup_max_lr": "预热结束学习率",
            "warmup_num_steps": "预热步数"
        },
        
        "zero_optimization": {
            "stage": "Zero优化级别：1(优化器状态分片), 2(+梯度分片), 3(+参数分片)",
            "allgather_partitions": "是否在allgather时分区",
            "allgather_bucket_size": "allgather桶大小",
            "overlap_comm": "计算通信重叠",
            "reduce_scatter": "使用reduce_scatter",
            "reduce_bucket_size": "reduce桶大小",
            "contiguous_gradients": "连续梯度存储"
        },
        
        "precision_settings": {
            "fp16.enabled": "启用FP16混合精度",
            "bf16.enabled": "启用BF16混合精度（推荐用于训练）",
            "loss_scale": "损失缩放：0为动态，>0为静态",
            "initial_scale_power": "初始缩放因子（2^power）",
            "loss_scale_window": "缩放因子更新窗口",
            "hysteresis": "缩放因子减少的迟滞"
        },
        
        "advanced_settings": {
            "gradient_clipping": "梯度裁剪阈值",
            "wall_clock_breakdown": "启用详细时间分析",
            "steps_per_print": "打印频率",
            "data_types.grad_accum_dtype": "梯度累积数据类型"
        }
    }
    
    for section, params in explanations.items():
        print(f"\n=== {section.upper().replace('_', ' ')} ===")
        for param, description in params.items():
            print(f"{param:30} : {description}")

if __name__ == "__main__":
    explain_config_sections()