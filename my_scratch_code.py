import torch
import torch.nn as nn

# 创建 MSELoss 实例
mse_loss_none = nn.MSELoss(reduction='none')
mse_loss_sum = nn.MSELoss(reduction='sum')
mse_loss_mean = nn.MSELoss(reduction='mean')

# 模拟真实值和预测值
y_true = torch.tensor([1.0, 2.0, 3.0])  # 真实值
y_pred = torch.tensor([1.5, 2.5, 2.8])  # 预测值

# 使用 reduction='none'
loss_none = mse_loss_none(y_pred, y_true)
print(f"Loss for each sample (reduction='none'): {loss_none}")

# 使用 reduction='sum'
loss_sum = mse_loss_sum(y_pred, y_true)
print(f"Sum of losses (reduction='sum'): {loss_sum.item()}")

# 使用 reduction='mean'
loss_mean = mse_loss_mean(y_pred, y_true)
print(f"Mean of losses (reduction='mean'): {loss_mean.item()}")
