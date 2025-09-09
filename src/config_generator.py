import json
import os

class ConfigGenerator:
    def __init__(self):
        self.base_config = {
            "train_batch_size": 384,
            "train_micro_batch_size_per_gpu": 16,
            "gradient_accumulation_steps": 4,
            "wall_clock_breakdown": False,
            "steps_per_print": 10
        }
    
    def generate_optimizer_configs(self):
        """生成不同优化器配置"""
        optimizers = {
            "adam": {
                "type": "Adam",
                "params": {
                    "lr": 0.001,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            "adamw": {
                "type": "AdamW", 
                "params": {
                    "lr": 0.0005,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 0.1
                }
            },
            "sgd": {
                "type": "SGD",
                "params": {
                    "lr": 0.01,
                    "momentum": 0.9,
                    "weight_decay": 0.0001
                }
            }
        }
        
        for name, optimizer in optimizers.items():
            config = self.base_config.copy()
            config["optimizer"] = optimizer
            config["zero_optimization"] = {"stage": 1}
            
            self.save_config(config, f"config/optimizer_{name}.json")
    
    def generate_precision_configs(self):
        """生成不同精度配置"""
        precisions = {
            "fp32": {
                "fp16": {"enabled": False},
                "bf16": {"enabled": False}
            },
            "fp16": {
                "fp16": {
                    "enabled": True,
                    "auto_cast": False,
                    "loss_scale": 0,
                    "initial_scale_power": 16,
                    "loss_scale_window": 1000,
                    "hysteresis": 2,
                    "min_loss_scale": 1
                }
            },
            "bf16": {
                "bf16": {"enabled": True}
            }
        }
        
        for name, precision in precisions.items():
            config = self.base_config.copy()
            config["optimizer"] = {
                "type": "Adam",
                "params": {"lr": 0.001, "weight_decay": 0.01}
            }
            config["zero_optimization"] = {"stage": 1}
            config.update(precision)
            
            self.save_config(config, f"config/precision_{name}.json")
    
    def generate_zero_configs(self):
        """生成不同Zero级别配置"""
        zero_configs = {
            "zero1": {
                "stage": 1,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },
            "zero2": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },
            "zero3": {
                "stage": 3,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9
            }
        }
        
        for name, zero_config in zero_configs.items():
            config = self.base_config.copy()
            config["optimizer"] = {
                "type": "Adam",
                "params": {"lr": 0.001, "weight_decay": 0.01}
            }
            config["zero_optimization"] = zero_config
            
            self.save_config(config, f"config/{name}.json")
    
    def save_config(self, config, filename):
        """保存配置文件"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Generated: {filename}")

def main():
    """生成所有配置文件"""
    generator = ConfigGenerator()
    
    print("Generating optimizer configurations...")
    generator.generate_optimizer_configs()
    
    print("\nGenerating precision configurations...")
    generator.generate_precision_configs()
    
    print("\nGenerating zero optimization configurations...")
    generator.generate_zero_configs()
    
    print("\nAll configurations generated successfully!")

if __name__ == "__main__":
    main()