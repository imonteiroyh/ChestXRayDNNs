import torch.nn.functional as F


def resize_4d_tensor(tensor, size):
    tensor = tensor.unsqueeze(0)

    resized_tensor = F.interpolate(tensor, size=size, mode="trilinear", align_corners=False)

    resized_tensor = resized_tensor.squeeze(0)

    return resized_tensor
