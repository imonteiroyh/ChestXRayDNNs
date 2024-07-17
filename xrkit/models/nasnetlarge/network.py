import timm
import torch
import torch.nn as nn

from xrkit.utilities.tensor import resize_4d_tensor


class NASNetLarge(nn.Module):
    def __init__(self, n_inputs: int = 3, pretrained=False):

        super().__init__()

        self.n_inputs = n_inputs
        self.network = timm.create_model("nasnetalarge", pretrained=pretrained, num_classes=10000)
        self.network.conv0.conv = nn.Conv2d(n_inputs, 96, kernel_size=3, stride=2, bias=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        original_sizes = tensor.size(2), tensor.size(3)

        tensor = self.network(tensor)
        tensor = tensor.unsqueeze(1).unsqueeze(2)
        tensor = resize_4d_tensor(tensor, size=(self.n_inputs, *original_sizes))

        return tensor
