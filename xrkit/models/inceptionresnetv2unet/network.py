import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class InceptionResNetV2UNet(nn.Module):
    def __init__(self, n_inputs=1, device="cuda"):
        super().__init__()

        self.features = {}
        self.device = device

        self.network = timm.create_model("inception_resnet_v2")
        self.network.conv2d_1a.conv = nn.Conv2d(n_inputs, 32, kernel_size=3, stride=2, bias=False)

        self.network.conv2d_2b.register_forward_hook(self.get_features(4))
        self.network.conv2d_4a.register_forward_hook(self.get_features(3))
        self.network.repeat.register_forward_hook(self.get_features(2))
        self.network.repeat_1.register_forward_hook(self.get_features(1))
        self.network.conv2d_7b.register_forward_hook(self.get_features(0))

        self.relu = nn.ReLU()

        self.conv1_1 = nn.Conv2d(2624, 256, kernel_size=3, padding="same")
        self.bn1_1 = nn.BatchNorm2d(256)
        self.conv1_2 = nn.Conv2d(256, 256, kernel_size=3, padding="same")
        self.bn1_2 = nn.BatchNorm2d(256)

        self.conv2_1 = nn.Conv2d(576, 128, kernel_size=3, padding="same")
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

    def forward(self, tensor):
        self.network(tensor)

        tensor = self.features[0]

        tensor = F.interpolate(tensor, size=(14, 14))
        tensor = torch.cat([tensor, self.features[1]], dim=1)

        tensor = self.conv1_1(tensor)
        tensor = self.bn1_1(tensor)
        tensor = self.relu(tensor)
        tensor = self.conv1_2(tensor)
        tensor = self.bn1_2(tensor)
        tensor = self.relu(tensor)

        tensor = F.interpolate(tensor, size=(29, 29))
        tensor = torch.cat([tensor, self.features[2]], dim=1)

        tensor = self.conv2_1(tensor)
        tensor = self.bn2_1(tensor)
        tensor = self.relu(tensor)
        tensor = self.conv2_2(tensor)
        tensor = self.bn2_2(tensor)
        tensor = self.relu(tensor)

        tensor = F.interpolate(tensor, size=(60, 60))
        tensor = torch.cat([tensor, self.features[3]], dim=1)

        tensor = self.conv3_1(tensor)
        tensor = self.bn3_1(tensor)
        tensor = self.relu(tensor)
        tensor = self.conv3_2(tensor)
        tensor = self.bn3_2(tensor)
        tensor = self.relu(tensor)

        tensor = F.interpolate(tensor, size=(125, 125))
        tensor = torch.cat([tensor, self.features[4]], dim=1)

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
    model = InceptionResNetV2UNet(device="cpu")

    print(model)

    print(summary(model, input_size, depth=4, device="cpu"))

    output = model(input)
    print(output.shape)
