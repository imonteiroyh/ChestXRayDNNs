import torch.nn as nn

from xrkit.models.resnet152v2.block import Bottleneck
from xrkit.utilities.tensor import resize_4d_tensor


class ResNet(nn.Module):
    def __init__(self, n_inputs, block, layer_sizes):
        super().__init__()

        self.n_inputs = n_inputs
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            n_inputs,
            self.inplanes,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.layer1 = self._make_layer(block, 64, layer_sizes[0])
        self.layer2 = self._make_layer(block, 128, layer_sizes[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer_sizes[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer_sizes[3], stride=2)
        self.avgpool = nn.AvgPool2d(5, stride=1)
        self.outplanes = 512 * block.expansion

    def _make_layer(self, block: nn.Module, planes: int, n_blocks: int, stride: int = 1) -> nn.Sequential:
        upsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, upsample)]
        self.inplanes = planes * block.expansion

        for _ in range(1, n_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, tensor):
        original_sizes = tensor.size(2), tensor.size(3)

        tensor = self.conv1(tensor)
        tensor = self.layer1(tensor)
        tensor = self.layer2(tensor)
        tensor = self.layer3(tensor)
        tensor = self.layer4(tensor)
        tensor = self.avgpool(tensor)

        tensor = resize_4d_tensor(tensor, size=(self.n_inputs, *original_sizes))

        return tensor


class ResNet152V2(ResNet):
    def __init__(self, n_inputs=1):
        super().__init__(n_inputs=n_inputs, block=Bottleneck, layer_sizes=[3, 8, 36, 3])
