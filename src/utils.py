import torch


def torch_circmean(x, min, max, *args, **kwargs):
    factor = (max - min)
    x = (x - min) / factor
    x = torch.atan2(
        torch.sin(x).mean(*args, **kwargs), 
        torch.cos(x).mean(*args, **kwargs), 
    )
    x = x * factor  + min
    return x