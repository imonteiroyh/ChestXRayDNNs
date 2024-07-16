import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes, stride=1, upsample=None):
        super().__init__()

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(
            outplanes,
            outplanes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn3 = nn.BatchNorm2d(outplanes)
        self.conv3 = nn.Conv2d(
            outplanes,
            outplanes * self.expansion,
            kernel_size=1,
            bias=False,
        )

        self.upsample = upsample

    def forward(self, tensor):
        if self.upsample is not None:
            residual = self.upsample(tensor)
        else:
            residual = tensor

        tensor = self.bn1(tensor)
        tensor = self.relu(tensor)
        tensor = self.conv1(tensor)

        tensor = self.bn2(tensor)
        tensor = self.relu(tensor)
        tensor = self.conv2(tensor)

        tensor = self.bn3(tensor)
        tensor = self.relu(tensor)
        tensor = self.conv3(tensor)

        tensor += residual

        return tensor
