import torch 

def collate_fn_for_None_type(batch):
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