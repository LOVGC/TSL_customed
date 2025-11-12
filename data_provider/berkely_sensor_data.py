import torch 
import numpy as np

def collate_fn_for_None_type(batch):
    # batch 是一个 list，每个元素是 (a, b, None)
    # 例如：[(a1, b1, None), (a2, b2, None), ...]

    transposed = list(zip(*batch))
    collated = []

    for items in transposed:
        # 如果整列全是 None
        if all(x is None for x in items):
            collated.append(None)
            continue

        # 把 numpy 转成 tensor
        items = [
            torch.as_tensor(x) if isinstance(x, np.ndarray) else x
            for x in items
        ]

        # 如果是 tensor，stack
        if isinstance(items[0], torch.Tensor):
            collated.append(torch.stack(items, dim=0))
        else:
            # 对其他类型（比如 int, str, list）保留原样
            collated.append(items)
    return tuple(collated)
