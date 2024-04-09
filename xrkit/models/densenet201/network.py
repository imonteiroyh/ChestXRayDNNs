import torch
import torch.nn as nn
from torchvision import models


class DenseNet201(nn.Module):
    def __init__(self, n_inputs: int = 1):
        """
        Class that modifies the architecture of DenseNet-201 to accommodate a specific number of input
        channels and to make the output have the same size of the input.

        Args:
            n_inputs (int, optional):
                The number of input channels of the image. Default is 1.
        """

        super().__init__()

        self.network = models.densenet201()
        self.network.features[0] = nn.Conv2d(n_inputs, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.network = nn.Sequential(*list(self.network.children())[:-1])

        self.conv1 = nn.Conv2d(1920, 16, kernel_size=1, padding=0, bias=False)
        self.up = nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False)
        self.conv2 = nn.Conv2d(16, n_inputs, kernel_size=1, padding=0, bias=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DenseNet201 model.

        Args:
            tensor (torch.Tensor):
                Input tensor.

        Returns:
            torch.Tensor:
                Output tensor.
        """

        tensor = self.network(tensor)
        tensor = self.conv1(tensor)
        tensor = self.up(tensor)
        tensor = self.conv2(tensor)

        return tensor
