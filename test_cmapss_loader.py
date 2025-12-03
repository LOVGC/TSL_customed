import os
import sys
import torch
from data_provider.data_factory import data_provider

# 创建一个简单的args类来模拟命令行参数
class Args:
    def __init__(self):
        self.data = 'C-MAPSS'
        self.root_path = 'd:\\GD\\TSL_customed\\dataset'
        self.data_path = 'train_FD001.txt'
        self.seq_len = 30
        self.label_len = 10
        self.pred_len = 5
        self.batch_size = 32
        self.freq = 'c'  # cycle
        self.embed = 'fixed'
        self.seasonal_patterns = 'Monthly'
        self.task_name = 'long_term_forecasting'
        self.features = 'MS'
        self.target = 'RUL'
        self.scale = True
        self.num_workers = 0

# 测试数据加载器
if __name__ == '__main__':
    args = Args()
    
    # 获取训练数据加载器
    train_data, train_loader = data_provider(args, 'train')
    print(f"训练数据大小: {len(train_data)}")
    
    # 获取验证数据加载器
    val_data, val_loader = data_provider(args, 'val')
    print(f"验证数据大小: {len(val_data)}")
    
    # 获取测试数据加载器
    test_data, test_loader = data_provider(args, 'test')
    print(f"测试数据大小: {len(test_data)}")
    
    # 尝试获取一个batch的数据
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"  batch_x shape: {batch_x.shape}")
        print(f"  batch_y shape: {batch_y.shape}")
        print(f"  batch_x_mark shape: {batch_x_mark.shape}")
        print(f"  batch_y_mark shape: {batch_y_mark.shape}")
        break  # 只显示第一个batch
        
    print("CMAPSS数据加载器测试成功！")