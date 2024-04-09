import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self) -> None:
        """
        Initializes the DiceBCELoss module.
        """

        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1) -> torch.Tensor:
        """
        Calculates the Dice loss combined with Binary Cross Entropy loss.

        Parameters:
            predictions (torch.Tensor):
                Predicted tensor.
            targets (torch.Tensor):
                Target tensor.
            smooth (float, optional):
                Smoothing factor to avoid division by zero. Defaults to 1.

        Returns:
            torch.Tensor:
                Combined Dice loss and Binary Cross Entropy loss.
        """

        predictions = F.sigmoid(predictions)

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
        bce = F.binary_cross_entropy(predictions, targets, reduction="mean")
        Dice_BCE = bce + dice_loss

        return Dice_BCE
