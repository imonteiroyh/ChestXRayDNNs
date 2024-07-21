import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class InceptionV3UNet(nn.Module):
    def __init__(self, n_inputs=1, device="cuda"):
        super().__init__()

        self.features = {}
        self.device = device

        self.network = timm.create_model("inception_v3")
        self.network.Conv2d_1a_3x3.conv = nn.Conv2d(n_inputs, 32, kernel_size=3, stride=2, bias=False)
        self.network = nn.Sequential(*list(self.network.children()))[:-3]

        self.network[2].register_forward_hook(self.get_features(3))
        self.network[5].register_forward_hook(self.get_features(2))
        self.network[10].branch3x3dbl_2.register_forward_hook(self.get_features(1))
        self.network[15].branch7x7x3_3.register_forward_hook(self.get_features(0))

        self.relu = nn.ReLU()

        self.conv1_1 = nn.Conv2d(2240, 256, kernel_size=3, padding="same")
        self.bn1_1 = nn.BatchNorm2d(256)
        self.conv1_2 = nn.Conv2d(256, 256, kernel_size=3, padding="same")
        self.bn1_2 = nn.BatchNorm2d(256)

        self.conv2_1 = nn.Conv2d(352, 128, kernel_size=3, padding="same")
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding="same")
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(320, 64, kernel_size=3, padding="same")
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, padding="same")
        self.bn3_2 = nn.BatchNorm2d(64)

        self.conv4_1 = nn.Conv2d(128, 32, kernel_size=3, padding="same")
        self.bn4_1 = nn.BatchNorm2d(32)
        self.conv4_2 = nn.Conv2d(32, 32, kernel_size=3, padding="same")
        self.bn4_2 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 1, kernel_size=1, padding="same")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:

        tensor = self.network(tensor)

        tensor = F.interpolate(tensor, size=(14, 14))
        tensor = torch.cat([tensor, self.features[0]], dim=1)

        tensor = self.conv1_1(tensor)
        tensor = self.bn1_1(tensor)
        tensor = self.relu(tensor)
        tensor = self.conv1_2(tensor)
        tensor = self.bn1_2(tensor)
        tensor = self.relu(tensor)

        tensor = F.interpolate(tensor, size=(29, 29))
        tensor = torch.cat([tensor, self.features[1]], dim=1)

        tensor = self.conv2_1(tensor)
        tensor = self.bn2_1(tensor)
        tensor = self.relu(tensor)
        tensor = self.conv2_2(tensor)
        tensor = self.bn2_2(tensor)
        tensor = self.relu(tensor)

        tensor = F.interpolate(tensor, size=(60, 60))
        tensor = torch.cat([tensor, self.features[2]], dim=1)

        tensor = self.conv3_1(tensor)
        tensor = self.bn3_1(tensor)
        tensor = self.relu(tensor)
        tensor = self.conv3_2(tensor)
        tensor = self.bn3_2(tensor)
        tensor = self.relu(tensor)

        tensor = F.interpolate(tensor, size=(125, 125))
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
    model = InceptionV3UNet(device="cpu")

    print(model)

    print(summary(model, input_size, depth=4, device="cpu"))

    output = model(input)
    print(output.shape)
