import torch
import torch.nn as nn

# 定义一个简单的 4D 输入张量 (N, C, H, W)
N, C, H, W = 2, 3, 4, 4  # 2 个样本，3 个通道，4x4 图像
x = torch.randn(N, C, H, W)  # 随机生成输入张量

# BatchNorm 手动计算
def manual_batchnorm(x, eps=1e-5):
    N, C, H, W = x.shape
    # 计算均值和方差，沿着 batch 和空间维度 (H, W) 计算
    mean = x.mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1) 
    var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)  # (1, C, 1, 1) 
    
    # 标准化
    x_normalized = (x - mean) / torch.sqrt(var + eps)  # (N, C, H, W)
    
    # 用于缩放和偏移（初始化为 1 和 0）
    gamma = torch.ones(1, C, 1, 1)
    beta = torch.zeros(1, C, 1, 1)
    
    # 缩放和偏移
    return gamma * x_normalized + beta

# LayerNorm 手动计算
def manual_layernorm(x, eps=1e-5):
    N, C, H, W = x.shape
    # 计算均值和方差，沿着所有特征维度 (C, H, W) 计算
    mean = x.mean(dim=(1, 2, 3), keepdim=True)  # (N, 1, 1, 1)
    var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)  # (N, 1, 1, 1)
    
    # 标准化
    x_normalized = (x - mean) / torch.sqrt(var + eps)  # (N, C, H, W)
    
    # 用于缩放和偏移（初始化为 1 和 0）
    gamma = torch.ones(1, C, 1, 1)
    beta = torch.zeros(1, C, 1, 1)
    
    # 缩放和偏移
    return gamma * x_normalized + beta

# InstanceNorm 手动计算
def manual_instancenorm(x, eps=1e-5):
    N, C, H, W = x.shape
    # 计算均值和方差，沿着空间维度 (H, W) 计算
    mean = x.mean(dim=(2, 3), keepdim=True)  # (N, C, 1, 1)
    var = x.var(dim=(2, 3), keepdim=True, unbiased=False)  # (N, C, 1, 1)
    
    # 标准化
    x_normalized = (x - mean) / torch.sqrt(var + eps)  # (N, C, H, W)
    
    # 用于缩放和偏移（初始化为 1 和 0）
    gamma = torch.ones(1, C, 1, 1)
    beta = torch.zeros(1, C, 1, 1)
    
    # 缩放和偏移
    return gamma * x_normalized + beta

# 使用 PyTorch 计算 BatchNorm, LayerNorm 和 InstanceNorm
batchnorm = nn.BatchNorm2d(C)
layernorm = nn.LayerNorm([C, H, W])  # LayerNorm 需要传入特征维度大小
instancenorm = nn.InstanceNorm2d(C)

# 应用 PyTorch 中的 BatchNorm, LayerNorm 和 InstanceNorm
x_pytorch = x.clone()  # 克隆原始输入，避免修改原数据

batchnorm_out = batchnorm(x_pytorch)
layernorm_out = layernorm(x_pytorch)
instancenorm_out = instancenorm(x_pytorch)

# 手动计算 BatchNorm、LayerNorm 和 InstanceNorm
batchnorm_manual_out = manual_batchnorm(x)
layernorm_manual_out = manual_layernorm(x)
instancenorm_manual_out = manual_instancenorm(x)

# 比较 PyTorch 和手动计算的结果
print("PyTorch BatchNorm 与手动 BatchNorm 是否一致：", torch.allclose(batchnorm_out, batchnorm_manual_out, atol=1e-6))
print("PyTorch LayerNorm 与手动 LayerNorm 是否一致：", torch.allclose(layernorm_out, layernorm_manual_out, atol=1e-6))
print("PyTorch InstanceNorm 与手动 InstanceNorm 是否一致：", torch.allclose(instancenorm_out, instancenorm_manual_out, atol=1e-6))
