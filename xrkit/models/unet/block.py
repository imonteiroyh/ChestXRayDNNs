from typing import Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, inplanes: int, outplanes: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.conv1(tensor)
        tensor = self.bn1(tensor)
        tensor = self.relu(tensor)
        tensor = self.conv2(tensor)
        tensor = self.bn2(tensor)
        tensor = self.relu(tensor)

        return tensor


class EncoderBlock(nn.Module):
    def __init__(self, inplanes: int, outplanes: int) -> None:
        super().__init__()

        self.conv = ConvBlock(inplanes, outplanes)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor = self.conv(tensor)
        pooled = self.pool(tensor)

        return tensor, pooled


class DecoderBlock(nn.Module):
    def __init__(self, inplanes: int, outplanes: int) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(inplanes, outplanes, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(outplanes + outplanes, outplanes)

    def forward(self, tensor: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        tensor = self.up(tensor)
        tensor = torch.cat([tensor, skip], dim=1)
        tensor = self.conv(tensor)

        return tensor
