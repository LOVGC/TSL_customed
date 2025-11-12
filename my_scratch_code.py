import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __getitem__(self, idx):
        a = torch.tensor([idx])             # 任意 tensor
        b = torch.tensor([idx * 10])        # 任意 tensor
        return a, None, b                   # 注意这里有 None

    def __len__(self):
        return 6

def my_collate_fn(batch):
    # batch 是一个 list，每个元素是 (a, b, None)
    # 例如：[(a1, b1, None), (a2, b2, None), ...]

    transposed = list(zip(*batch))  # [(a1,a2,...), (b1,b2,...), (None,None,...)]
    collated = []

    for items in transposed:
        # 如果该列全是 None，就保持 None
        if all(x is None for x in items):
            collated.append(None)
        # 否则用默认方式堆叠
        else:
            collated.append(torch.stack(items, dim=0))
    return tuple(collated)

dataset = MyDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=my_collate_fn)

for a, b, c in loader:
    print(a, b, c)
