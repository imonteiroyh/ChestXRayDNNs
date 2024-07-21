from typing import Any, Callable, Dict, Iterable, Tuple

import pytorch_lightning as L
import torch

from xrkit.models.inceptionresnetv2unet import InceptionResNetV2UNet
from xrkit.models.lightning.base import BaseModel
from xrkit.segmentation import (
    DiceBCELoss,
    average_surface_distance,
    balanced_average_hausdorff_distance,
    dice,
    jaccard_index,
)

# mypy: disable-error-code="misc"


class InceptionResNetV2UNetModel(L.LightningModule, BaseModel):
    def __init__(self, n_epochs: int, **kwargs) -> None:
        super().__init__()

        network = InceptionResNetV2UNet(n_inputs=1, **kwargs)
        criterion = DiceBCELoss()
        metrics: Iterable[Tuple[Callable, Dict[str, Any]]] = (
            (dice, {}),
            (jaccard_index, {}),
            (balanced_average_hausdorff_distance, {}),
            (average_surface_distance, {}),
        )
        activation_function = torch.sigmoid
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        self.save_hyperparameters()
        self.setup_model(
            network=network,
            criterion=criterion,
            metrics=metrics,
            activation_function=activation_function,
            optimizer=optimizer,
            scheduler=scheduler,
        )


if __name__ == "__main__":
    input = torch.rand((4, 1, 256, 256))

    model = InceptionResNetV2UNetModel(n_epochs=1).network
    print(model(input).shape)
