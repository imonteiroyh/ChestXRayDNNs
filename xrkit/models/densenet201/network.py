import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torchvision import models


class DenseNet201(nn.Module):
    def __init__(self, n_inputs=3, n_outputs=13, pretrained = True, device="cuda"):
        super().__init__()

        self.features = {}
        self.device = device

        weights = None if not pretrained else models.DenseNet201_Weights.DEFAULT
        self.network = models.densenet201(weights=weights)
        for parameter in self.network.parameters():
            parameter.requires_grad = False

        self.network.classifier = nn.Linear(1920, n_outputs, bias=True)

    def forward(self, tensor):
        return self.network(tensor)

if __name__ == "__main__":
    input_size = (4, 3, 256, 256)
    input = torch.rand(input_size)
    model = DenseNet201(device="cpu")

    print(model)

    print(summary(model, input_size, depth=4, device="cpu"))

    output = model(input)
    print(output.shape)
