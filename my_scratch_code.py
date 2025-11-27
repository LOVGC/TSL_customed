import torch
import torch.nn.functional as F

# 1. 模型的输出 logits（未经过 softmax）
logits = torch.tensor([2.0, 1.0, 0.1])

# 2. 目标标签（真实类别）
target = torch.tensor([0])  # 目标是类别 0

# 手动计算 softmax 转换成概率
exp_logits = torch.exp(logits)  # 对每个 logit 取指数
softmax_probs = exp_logits / exp_logits.sum()  # 归一化为概率分布

# 手动计算交叉熵损失
# 交叉熵损失是 -log(P(target))
manual_loss = -torch.log(softmax_probs[target])  # 获取目标类别的预测概率并计算 log

print(f"手动计算的交叉熵损失: {manual_loss.item()}")

# 3. 使用 PyTorch 的 CrossEntropyLoss 计算损失
criterion = torch.nn.CrossEntropyLoss()
pytorch_loss = criterion(logits.unsqueeze(0), target)  # logits 需要是 [batch_size, num_classes]

print(f"PyTorch 计算的交叉熵损失: {pytorch_loss.item()}")
