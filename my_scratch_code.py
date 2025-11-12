import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __getitem__(self, idx):
        a = torch.tensor([idx])             # 任意 tensor
        b = torch.tensor([idx * 10])        # 任意 tensor
        return a, None, b                   # 注意这里有 None

    def __len__(self):
        return 5

dataset = MyDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=False)

for batch in loader:
    print(batch)
    print(f"type: {[type(x) for x in batch]}")
