
import deepspeed
import torch.nn as nn
import torch
from torch.utils.data import DataLoader,TensorDataset
import json
import argparse
import time
import os

class TestModel(nn.Module):

    def __init__(self, input_size = 1024, hidden_size = 2068, num_classes = 1000):
        super(TestModel,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size,num_classes)
        )


def create_test_data(batch_size=576, num_batches=50):
    """创建测试数据"""
    X = torch.randn(batch_size * num_batches, 1024)
    y = torch.randint(0, 1000, (batch_size * num_batches,))
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def test_config(config_path, epochs=2):
    """测试特定配置"""
    print(f"\n{'='*50}")
    print(f"Testing config: {config_path}")
    print(f"{'='*50}")
    
    # 读取配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Config summary:")
    print(f"  Batch size: {config.get('train_batch_size', 'N/A')}")
    print(f"  Micro batch: {config.get('train_micro_batch_size_per_gpu', 'N/A')}")
    print(f"  Optimizer: {config.get('optimizer', {}).get('type', 'N/A')}")
    print(f"  FP16: {config.get('fp16', {}).get('enabled', False)}")
    print(f"  BF16: {config.get('bf16', {}).get('enabled', False)}")
    print(f"  Zero stage: {config.get('zero_optimization', {}).get('stage', 'N/A')}")
    
    # 创建参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--deepspeed_config', type=str, default=config_path)
    args = parser.parse_args(['--deepspeed_config', config_path])
    
    # 创建模型和数据
    model = TestModel()
    dataloader = create_test_data(batch_size=576)
    criterion = nn.CrossEntropyLoss()
    
    # DeepSpeed初始化
    try:
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=model.parameters()
        )
        
        # 记录内存使用
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        
        # 训练
        model_engine.train()
        start_time = time.time()
        total_loss = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (data, target) in enumerate(dataloader):
                if batch_idx >= 10:  # 限制批次数量以节省时间
                    break
                    
                data = data.to(model_engine.local_rank)
                target = target.to(model_engine.local_rank)
                
                outputs = model_engine(data)
                loss = criterion(outputs, target)
                
                model_engine.backward(loss)
                model_engine.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 5 == 0:
                    print(f'  Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            total_loss += epoch_loss
            print(f'  Epoch {epoch} avg loss: {epoch_loss/min(10, len(dataloader)):.4f}')
        
        end_time = time.time()
        
        # 记录结果
        results = {
            'config': config_path,
            'training_time': end_time - start_time,
            'avg_loss': total_loss / (epochs * min(10, len(dataloader))),
            'success': True
        }
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            results['peak_memory_mb'] = peak_memory / 1024 / 1024
            print(f"  Peak GPU memory: {peak_memory/1024/1024:.1f} MB")
        
        print(f"  Training time: {end_time - start_time:.2f} seconds")
        print(f"  Average loss: {results['avg_loss']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return {
            'config': config_path,
            'error': str(e),
            'success': False
        }
    

def main():
    """测试所有配置"""
    configs = [
        '../configs/basic_config.json',
        # '../configs/fp16_config.json',
        # '../configs/advanced_config.json'
    ]
    
    results = []
    
    for config_path in configs:
        if os.path.exists(config_path):
            result = test_config(config_path)
            results.append(result)
        else:
            print(f"Config file not found: {config_path}")
    
    # 总结结果
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL TESTS")
    print(f"{'='*60}")
    
    for result in results:
        print(f"\nConfig: {result['config']}")
        if result['success']:
            print(f"  Status: SUCCESS")
            print(f"  Time: {result['training_time']:.2f}s")
            print(f"  Loss: {result['avg_loss']:.4f}")
            if 'peak_memory_mb' in result:
                print(f"  Memory: {result['peak_memory_mb']:.1f}MB")
        else:
            print(f"  Status: FAILED")
            print(f"  Error: {result['error']}")
    
    # 保存结果
    with open('config_test_logs.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: config_test_logs.json")

if __name__ == "__main__":
    main()

