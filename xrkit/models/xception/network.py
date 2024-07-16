import timm
import torch
import torch.nn as nn

from xrkit.utilities.tensor import resize_4d_tensor


class Xception(nn.Module):
    def __init__(self, n_inputs: int = 1, pretrained=False):

        super().__init__()

        self.n_inputs = n_inputs
        self.network = timm.create_model("legacy_xception", pretrained=pretrained, num_classes=10000)
        self.network.conv1 = nn.Conv2d(n_inputs, 32, kernel_size=3, stride=2, bias=False)
        self.network = nn.Sequential(*list(self.network.children()))[:-1]

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        original_sizes = tensor.size(2), tensor.size(3)

        tensor = self.network(tensor)
        tensor = tensor.unsqueeze(1).unsqueeze(2)
        tensor = resize_4d_tensor(tensor, size=(self.n_inputs, *original_sizes))

        return tensor


if __name__ == "__main__":
    input = torch.rand((4, 1, 256, 256))
    output = Xception()(input)

    print(output.shape)
