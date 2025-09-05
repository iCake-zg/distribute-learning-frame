import torch
import deepspeed
import sys

def test_installation():
    print("=== DeepSpeed Installation Test ===")
    
    # 检查版本
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"DeepSpeed version: {deepspeed.__version__}")
    
    # 检查CUDA支持
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # 测试基本导入
    try:
        from deepspeed import initialize
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
        print("✅ DeepSpeed core modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # 创建简单模型测试
    try:
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # 基本的DeepSpeed配置
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {"lr": 0.01}
            }
        }
        
        print("✅ Basic configuration created successfully")
        print("🎉 DeepSpeed installation test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating model/config: {e}")
        return False

if __name__ == "__main__":
    success = test_installation()
    exit(0 if success else 1)