import torch
import torch.nn as nn
from torchvision import models

from xrkit.utilities.tensor import resize_4d_tensor

# mypy: disable-error-code="misc"


class DenseNet201(nn.Module):
    def __init__(self, task: str, n_outputs: int = 1, pretrained: bool = False):
        super().__init__()

        self.task = task
        self.n_outputs = n_outputs
        self.network = models.densenet201()

        if self.task == "segmentation":
            self.network = nn.Sequential(*list(self.network.children())[:-1])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        original_sizes = tensor.size(1), tensor.size(2), tensor.size(3)
    
        tensor = self.network(tensor)
        
        if self.task == 'segmentation':
            tensor = resize_4d_tensor(tensor, size=(*original_sizes,))
    
        return tensor
