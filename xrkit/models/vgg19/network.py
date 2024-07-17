import torch
import torch.nn as nn
from torchvision import models

from xrkit.utilities.tensor import resize_4d_tensor


class VGG19(nn.Module):
    def __init__(self, n_inputs: int = 3):

        super().__init__()

        self.n_inputs = n_inputs
        self.network = models.vgg19()
        self.network.features[0] = nn.Conv2d(n_inputs, 64, kernel_size=3, stride=1, padding=1)
        self.network = nn.Sequential(*list(self.network.children()))[:-1]

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        original_sizes = tensor.size(2), tensor.size(3)

        tensor = self.network(tensor)

        tensor = resize_4d_tensor(tensor, size=(self.n_inputs, *original_sizes))

        return tensor
