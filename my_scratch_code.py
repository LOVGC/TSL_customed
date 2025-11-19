import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# -------------------------------
# 示例 1：__getitem__ 返回 tuple (tensor, int)
# -------------------------------
class TupleDataset(Dataset):
    def __init__(self, n):
        self.n = n
        # 构造一些可变长度的序列 (1D tensor)，长度从 1 到 n
        self.data = [torch.arange(i + 1, dtype=torch.float32) for i in range(n)]
        self.labels = list(range(n))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def collate_tuple(batch):
    """
    batch 是 [(tensor, int), (tensor, int), ...]
    我们 pad 第一个元素 (tensor)，把 labels 合并成一个 tensor
    """
    sequences, labels = zip(*batch)  # unzip
    # pad_sequence 会把不同长度的 sequence pad 成一个 batch tensor
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, labels


# -------------------------------
# 示例 2：__getitem__ 返回 dict
# -------------------------------
class DictDataset(Dataset):
    def __init__(self, n):
        self.n = n
        # 用字典来表示数据，比如 “feature” 是张量，“meta” 是一个 float
        self.features = [torch.randn(3) for _ in range(n)]
        self.meta = [float(i) * 0.1 for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"feature": self.features[idx], "meta": self.meta[idx]}


def collate_dict(batch):
    """
    batch 是 [{feature: tensor, meta: float}, {...}, ...]
    我们把 feature 拼成一个 batch tensor，把 meta 拼成 tensor
    """
    features = [d["feature"] for d in batch]
    metas = [d["meta"] for d in batch]
    features = torch.stack(features, dim=0)
    metas = torch.tensor(metas, dtype=torch.float32)
    return {"feature": features, "meta": metas}


# -------------------------------
# 示例 3：__getitem__ 返回自定义 class
# -------------------------------
class MySample:
    def __init__(self, x, y):
        self.x = x  # torch.Tensor
        self.y = y  # float

class ClassDataset(Dataset):
    def __init__(self, n):
        self.n = n
        self.samples = [MySample(torch.randn(2), float(i)) for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_class(batch):
    """
    batch 是 [MySample, MySample, ...]
    我们提取 .x 和 .y，拼成 batch。
    最后返回一个自定义容器 (这里还是 dict，但你也可以返回列表、命名元组、自定义 class 等)
    """
    xs = torch.stack([s.x for s in batch], dim=0)
    ys = torch.tensor([s.y for s in batch], dtype=torch.float32)
    return MySample(x=xs, y=ys)  # 注意：这里构造的 MySample.x 是 (batch, *) 维度


# -------------------------------
# 验证 DataLoader
# -------------------------------
if __name__ == "__main__":
    # 示例 1
    ds1 = TupleDataset(5)
    loader1 = DataLoader(ds1, batch_size=2, collate_fn=collate_tuple)
    for batch in loader1:
        padded, labels = batch
        print("TupleDataset batch:")
        print("  padded:", padded, "shape:", padded.shape)
        print("  labels:", labels, "shape:", labels.shape)
        break

    # 示例 2
    ds2 = DictDataset(5)
    loader2 = DataLoader(ds2, batch_size=2, collate_fn=collate_dict)
    for batch in loader2:
        print("DictDataset batch:")
        print("  feature:", batch["feature"], "shape:", batch["feature"].shape)
        print("  meta:", batch["meta"], "shape:", batch["meta"].shape)
        break

    # 示例 3
    ds3 = ClassDataset(5)
    loader3 = DataLoader(ds3, batch_size=2, collate_fn=collate_class)
    for batch in loader3:
        print("ClassDataset batch:")
        print("  x:", batch.x, "shape:", batch.x.shape)
        print("  y:", batch.y, "shape:", batch.y.shape)
        break
