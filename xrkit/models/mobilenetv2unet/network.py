import torch
import torch.nn as nn
from torchinfo import summary
from torchvision import models


class MobileNetV2UNet(nn.Module):
    def __init__(self, n_inputs=1, device="cuda"):
        super().__init__()

        self.features = {}
        self.device = device

        self.network = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.network.features[0] = nn.Conv2d(n_inputs, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.network = nn.Sequential(*list(self.network.children()))[:-1]

        self.network[0][2].conv[0].register_forward_hook(self.get_features("128"))
        self.network[0][4].conv[0].register_forward_hook(self.get_features("64"))
        self.network[0][7].conv[0].register_forward_hook(self.get_features("32"))
        self.network[0][14].conv[0].register_forward_hook(self.get_features("16"))
        self.feature_order = ["16", "32", "64", "128"]

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv = nn.Conv2d(32, 1, kernel_size=1, padding="same")

        self.make_layers()

    def make_layers(self, in_channels=[1856, 448, 272, 160], out_channels=[256, 128, 64, 32]):
        if len(in_channels) != len(out_channels):
            raise ValueError("in_channels and out_channels must have the same length.")

        self.blocks = []
        for index in range(len(in_channels)):
            block = nn.Sequential(
                nn.Conv2d(in_channels[index], out_channels[index], kernel_size=3, padding="same"),
                nn.BatchNorm2d(out_channels[index]),
                nn.ReLU(),
                nn.Conv2d(out_channels[index], out_channels[index], kernel_size=3, padding="same"),
                nn.BatchNorm2d(out_channels[index]),
                nn.ReLU(),
            ).to(self.device)

            self.blocks.append(block)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:

        tensor = self.network(tensor)

        # print([i.shape for i in self.features.values()])
        for index, block in enumerate(self.blocks):
            tensor = self.upsample(tensor)
            tensor = torch.cat([tensor, self.features[self.feature_order[index]]], dim=1)
            tensor = block(tensor)

        tensor = self.upsample(tensor)
        tensor = self.conv(tensor)

        return tensor

    def get_features(self, name):
        def hook(model, input, output):
            self.features[name] = output

        return hook


if __name__ == "__main__":
    input_size = (4, 3, 256, 256)
    input = torch.rand(input_size)
    model = MobileNetV2UNet(device="cpu")

    print(model)

    print(summary(model, input_size, depth=4, device="cpu"))

    output = model(input)
    print(output.shape)