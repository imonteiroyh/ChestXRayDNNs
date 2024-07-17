import torch
import torch.nn as nn
from torchinfo import summary
from torchvision import models


class DenseNet201(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = models.densenet201()
        # self.network = nn.Sequential(*list(self.network.children())[:-1])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.network(tensor)

        return tensor


if __name__ == "__main__":
    input_size = (4, 3, 256, 256)
    input = torch.rand(input_size).to("cuda")
    model = DenseNet201().to("cuda")

    print(model)

    print(summary(model, input_size))

    output = model(input)
    print(output.shape)
