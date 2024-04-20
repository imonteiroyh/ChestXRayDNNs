import torch
import torch.nn as nn
from torchvision import models

from xrkit.utilities.tensor import resize_4d_tensor

# mypy: disable-error-code="misc"


class InceptionV3(nn.Module):
    def __init__(self, task: str, n_inputs: int = 1, n_outputs: int = 1, pretrained: bool = False):

        super().__init__()

        self.task = task
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.network = models.inception_v3()
        self.network.Conv2d_1a_3x3.conv = nn.Conv2d(n_inputs, 32, kernel_size=3, stride=2, bias=False)

        task_map = {"segmentation": self.__segmentation_forward}

        self.task_forward = task_map.get(self.task)

        if self.task_forward is None:
            raise ValueError("Invalid task.")

    def __segmentation_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        original_sizes = tensor.size(2), tensor.size(3)

        tensor = resize_4d_tensor(tensor, size=(self.n_inputs, 300, 300))
        inception_output = self.network(tensor)
        tensor = inception_output.logits

        tensor = tensor.unsqueeze(1).unsqueeze(2)
        tensor = resize_4d_tensor(tensor, size=(self.n_outputs, *original_sizes))

        return tensor

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.task_forward(tensor)

        return tensor
