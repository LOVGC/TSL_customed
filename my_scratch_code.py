import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # Windows
plt.rcParams["axes.unicode_minus"] = False

# Epoch 1-29
epochs = list(range(1, 30))

# Extracted from your log
val_acc = [
    0.844, 0.840, 0.904, 0.904, 0.919, 0.894, 0.912, 0.924, 0.916,
    0.920, 0.920, 0.923, 0.914, 0.935, 0.929, 0.930, 0.927, 0.919,
    0.937, 0.932, 0.935, 0.931, 0.933, 0.929, 0.934, 0.917, 0.925,
    0.931, 0.931
]

test_acc = [
    0.838, 0.848, 0.913, 0.910, 0.909, 0.894, 0.917, 0.935, 0.926,
    0.928, 0.929, 0.932, 0.923, 0.934, 0.935, 0.932, 0.933, 0.924,
    0.935, 0.929, 0.934, 0.934, 0.938, 0.939, 0.935, 0.917, 0.928,
    0.931, 0.935
]

# 图 1：验证集准确率
plt.figure()
plt.plot(epochs, val_acc, marker="o")
plt.xlabel("训练轮次（Epoch）")
plt.ylabel("验证集准确率")
plt.title("验证集准确率随训练轮次变化")
plt.xticks(epochs)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

# 图 2：测试集准确率
plt.figure()
plt.plot(epochs, test_acc, marker="o")
plt.xlabel("训练轮次（Epoch）")
plt.ylabel("测试集准确率")
plt.title("测试集准确率随训练轮次变化")
plt.xticks(epochs)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()
