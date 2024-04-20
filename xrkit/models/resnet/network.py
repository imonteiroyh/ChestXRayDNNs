import timm
import torch
import torch.nn as nn

from xrkit.utilities.tensor import resize_4d_tensor

# mypy: disable-error-code="misc"


class ResNet152V2(nn.Module):
    def __init__(self, task: str, n_inputs: int = 1, n_outputs: int = 1, pretrained: bool = False):
        super().__init__()

        self.task = task
        self.n_outputs = n_outputs

        num_classes = 10000 if self.task == "segmentation" else n_outputs

        self.network = timm.create_model("resnet152d", pretrained=pretrained, num_classes=num_classes)
        self.network.conv1[0] = nn.Conv2d(n_inputs, 32, kernel_size=3, stride=2, padding=1, bias=False)

        if self.task == "segmentation":
            self.network = nn.Sequential(*list(self.network.children()))[:-2]

        task_map = {"segmentation": self.__segmentation_forward}

        self.task_forward = task_map.get(self.task)

        if self.task_forward is None:
            raise ValueError("Invalid task.")

    def __segmentation_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        original_sizes = tensor.size(2), tensor.size(3)

        tensor = self.network(tensor)

        tensor = resize_4d_tensor(tensor, size=(self.n_outputs, *original_sizes))

        return tensor

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.task_forward(tensor)

        return tensor
