from typing import Tuple

import torch
import torch.nn.functional as F


def resize_4d_tensor(tensor: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
    """
    Resize a 4D tensor to a specified size using trilinear interpolation.

    Args:
        tensor (torch.Tensor):
            Input tensor with shape (batch_size, channels, depth, height, width).
        size (Tuple[int, int, int]):
            Desired size of the output tensor in the format (depth, height, width).

    Returns:
        torch.Tensor:
            Resized tensor with the specified size.
    """

    tensor = tensor.unsqueeze(0)

    resized_tensor = F.interpolate(tensor, size=size, mode="trilinear", align_corners=False)

    resized_tensor = resized_tensor.squeeze(0)

    return resized_tensor
