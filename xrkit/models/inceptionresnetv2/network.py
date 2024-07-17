import timm
import torch
import torch.nn as nn

from xrkit.utilities.tensor import resize_4d_tensor


class InceptionResNetV2(nn.Module):
    def __init__(self, n_inputs: int = 3, pretrained=False):

        super().__init__()

        self.n_inputs = n_inputs
        self.network = timm.create_model("inception_resnet_v2", pretrained=pretrained, num_classes=10000)
        breakpoint()
        self.network.conv2d_1a.conv = nn.Conv2d(n_inputs, 32, kernel_size=3, stride=2, bias=False)
        self.network = nn.Sequential(*list(self.network.children()))[:-1]

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        original_sizes = tensor.size(2), tensor.size(3)

        tensor = self.network(tensor)
        tensor = tensor.unsqueeze(1).unsqueeze(2)
        tensor = resize_4d_tensor(tensor, size=(self.n_inputs, *original_sizes))

        return tensor


if __name__ == "__main__":
    input = torch.rand((4, 3, 256, 256))
    output = InceptionResNetV2()(input)

    print(output.shape)
