import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class NASNetLargeUNet(nn.Module):
    def __init__(self, n_inputs=1, device="cuda"):
        super().__init__()

        self.features = {}
        self.device = device

        self.network = timm.create_model("nasnetalarge")
        self.network.conv0.conv = nn.Conv2d(n_inputs, 96, kernel_size=3, stride=2, bias=False)
        #[a for a, _ in self.network.named_children()]

        self.network.cell_stem_0.comb_iter_0_left.act_1.register_forward_hook(self.get_features(4))
        self.network.cell_stem_1.comb_iter_0_left.act_1.register_forward_hook(self.get_features(3))
        self.network.reduction_cell_0.comb_iter_0_left.act_1.register_forward_hook(self.get_features(2))
        self.network.reduction_cell_1.comb_iter_0_left.act_1.register_forward_hook(self.get_features(1))
        self.network.act.register_forward_hook(self.get_features(0))

        self.relu = nn.ReLU()

        self.conv1_1 = nn.Conv2d(4704, 256, kernel_size=3, padding="same")
        self.bn1_1 = nn.BatchNorm2d(256)
        self.conv1_2 = nn.Conv2d(256, 256, kernel_size=3, padding="same")
        self.bn1_2 = nn.BatchNorm2d(256)

        self.conv2_1 = nn.Conv2d(592, 128, kernel_size=3, padding="same")
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding="same")
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(212, 64, kernel_size=3, padding="same")
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, padding="same")
        self.bn3_2 = nn.BatchNorm2d(64)

        self.conv4_1 = nn.Conv2d(106, 32, kernel_size=3, padding="same")
        self.bn4_1 = nn.BatchNorm2d(32)
        self.conv4_2 = nn.Conv2d(32, 32, kernel_size=3, padding="same")
        self.bn4_2 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 1, kernel_size=1, padding="same")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        self.network(tensor)

        tensor = self.features[0]

        tensor = F.interpolate(tensor, size=(16, 16))
        tensor = torch.cat([tensor, self.features[1]], dim=1)

        tensor = self.conv1_1(tensor)
        tensor = self.bn1_1(tensor)
        tensor = self.relu(tensor)
        tensor = self.conv1_2(tensor)
        tensor = self.bn1_2(tensor)
        tensor = self.relu(tensor)

        tensor = F.interpolate(tensor, size=(32, 32))
        tensor = torch.cat([tensor, self.features[2]], dim=1)

        tensor = self.conv2_1(tensor)
        tensor = self.bn2_1(tensor)
        tensor = self.relu(tensor)
        tensor = self.conv2_2(tensor)
        tensor = self.bn2_2(tensor)
        tensor = self.relu(tensor)

        tensor = F.interpolate(tensor, size=(64, 64))
        tensor = torch.cat([tensor, self.features[3]], dim=1)

        tensor = self.conv3_1(tensor)
        tensor = self.bn3_1(tensor)
        tensor = self.relu(tensor)
        tensor = self.conv3_2(tensor)
        tensor = self.bn3_2(tensor)
        tensor = self.relu(tensor)

        tensor = F.interpolate(tensor, size=(127, 127))
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
    model = NASNetLargeUNet(device="cpu")

    print(model)

    print(summary(model, input_size, depth=4, device="cpu"))

    output = model(input)
    print(output.shape)
