import torch


def moving_average(holder: torch.Tensor, input_tensor: torch.Tensor, average_ratio: float) -> torch.Tensor:
    assert input_tensor.dim() == 2
    with torch.no_grad():
        holder = holder * (1 - average_ratio) + average_ratio * input_tensor.mean(0).view(-1)
    return holder
