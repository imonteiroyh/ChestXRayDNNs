import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torchvision import models


class MobileNetV2UNet(nn.Module):
    def __init__(self, n_inputs=1, device="cuda"):
        super().__init__()

        self.features = {}
        self.device = device

        self.network = models.mobilenet_v2()
        self.network.features[0][0] = nn.Conv2d(n_inputs, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.network = nn.Sequential(*list(self.network.children()))[:-1]

        self.network[0][2].conv[0].register_forward_hook(self.get_features(3))
        self.network[0][4].conv[0].register_forward_hook(self.get_features(2))
        self.network[0][7].conv[0].register_forward_hook(self.get_features(1))
        self.network[0][14].conv[0].register_forward_hook(self.get_features(0))

        self.relu = nn.ReLU()

        self.conv1_1 = nn.Conv2d(1856, 256, kernel_size=3, padding="same")
        self.bn1_1 = nn.BatchNorm2d(256)
        self.conv1_2 = nn.Conv2d(256, 256, kernel_size=3, padding="same")
        self.bn1_2 = nn.BatchNorm2d(256)

        self.conv2_1 = nn.Conv2d(448, 128, kernel_size=3, padding="same")
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding="same")
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(272, 64, kernel_size=3, padding="same")
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, padding="same")
        self.bn3_2 = nn.BatchNorm2d(64)

        self.conv4_1 = nn.Conv2d(160, 32, kernel_size=3, padding="same")
        self.bn4_1 = nn.BatchNorm2d(32)
        self.conv4_2 = nn.Conv2d(32, 32, kernel_size=3, padding="same")
        self.bn4_2 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 1, kernel_size=1, padding="same")

    def forward(self, tensor):

        tensor = self.network(tensor)

        tensor = F.interpolate(tensor, size=(16, 16))
        tensor = torch.cat([tensor, self.features[0]], dim=1)

        tensor = self.conv1_1(tensor)
        tensor = self.bn1_1(tensor)
        tensor = self.relu(tensor)
        tensor = self.conv1_2(tensor)
        tensor = self.bn1_2(tensor)
        tensor = self.relu(tensor)

        tensor = F.interpolate(tensor, size=(32, 32))
        tensor = torch.cat([tensor, self.features[1]], dim=1)

        tensor = self.conv2_1(tensor)
        tensor = self.bn2_1(tensor)
        tensor = self.relu(tensor)
        tensor = self.conv2_2(tensor)
        tensor = self.bn2_2(tensor)
        tensor = self.relu(tensor)

        tensor = F.interpolate(tensor, size=(64, 64))
        tensor = torch.cat([tensor, self.features[2]], dim=1)

        tensor = self.conv3_1(tensor)
        tensor = self.bn3_1(tensor)
        tensor = self.relu(tensor)
        tensor = self.conv3_2(tensor)
        tensor = self.bn3_2(tensor)
        tensor = self.relu(tensor)

        tensor = F.interpolate(tensor, size=(128, 128))
        tensor = torch.cat([tensor, self.features[3]], dim=1)

        tensor = self.conv4_1(tensor)
        tensor = self.bn4_1(tensor)
        tensor = self.relu(tensor)
        tensor = self.conv4_2(tensor)
        tensor = self.bn4_2(tensor)
        tensor = self.relu(tensor)

        tensor = F.interpolate(tensor, size=(256, 256))
        tensor = self.conv5(tensor)

        return tensor

    def get_features(self, id):
        def hook(_, __, output):
            self.features[id] = output

        return hook


if __name__ == "__main__":
    input_size = (4, 1, 256, 256)
    input = torch.rand(input_size)
    model = MobileNetV2UNet(device="cpu")

    print(model)

    print(summary(model, input_size, depth=4, device="cpu"))

    output = model(input)
    print(output.shape)
