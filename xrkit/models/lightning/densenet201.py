from typing import Any, Callable, Dict, Iterable, Tuple

import pytorch_lightning as L
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy, auroc, f1_score, fbeta_score, precision, recall, specificity

from xrkit.models.densenet201 import DenseNet201
from xrkit.models.lightning.base import BaseModel

# mypy: disable-error-code="misc"


class DenseNet201Model(L.LightningModule, BaseModel):
    def __init__(self, n_epochs: int, n_inputs=1, n_outputs=13, **kwargs) -> None:
        super().__init__()

        network = DenseNet201(n_inputs=1, **kwargs)
        criterion = nn.BCEWithLogitsLoss()
        metrics = (
            (f1_score, {"num_labels": n_outputs, "task": "multilabel"}),
            (accuracy, {"num_labels": n_outputs, "task": "multilabel"}),
            (recall, {"num_labels": n_outputs, "task": "multilabel"}),
            (precision, {"num_labels": n_outputs, "task": "multilabel"}),
            (specificity, {"num_labels": n_outputs, "task": "multilabel"}),
            # (auroc, {"num_labels": n_outputs, "task": "multilabel"}),
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
    input = torch.rand((4, 3, 256, 256))

    model = DenseNet201Model(n_epochs=1, device="cpu").network
    print(model(input).shape)

    model = DenseNet201Model(n_epochs=1, pretrained=True, device="cpu").network
    print(model(input).shape)
