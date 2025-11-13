import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取 CSV 文件
file_path = r"dataset\ETT-small\ETTh1.csv"
data = pd.read_csv(file_path)

# 2. 打印所有可选的列名
print("Available columns:", list(data.columns))

# 3. 用户输入想要绘制的列名
column_name = input("请输入要绘制的列名（例如 OT）: ").strip()

# 4. 检查列名是否存在
if column_name not in data.columns:
    print(f"❌ 列名 '{column_name}' 不存在，请重新运行程序并输入正确的列名。")
else:
    # 5. 提取数据并绘图
    date = pd.to_datetime(data['date'])
    values = data[column_name]

    plt.figure(figsize=(12, 5))
    plt.plot(date, values, label=column_name, color='b')
    plt.xlabel('Date')
    plt.ylabel(column_name)
    plt.title(f'{column_name} over Time (ETTh1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
