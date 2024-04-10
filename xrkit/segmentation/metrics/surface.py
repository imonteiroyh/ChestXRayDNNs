import torch
from medpy.metric.binary import asd

from xrkit.utilities.array import handle_empty_array


def average_surface_distance(
    predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, connectivity: int = 1
) -> torch.Tensor:
    """
    Calculates the average surface distance between two binary segmentation masks.

    Parameters:
        predictions (torch.Tensor):
            The first binary segmentation mask tensor.
        targets (torch.Tensor):
            The second binary segmentation mask tensor.
        threshold (float, optional):
            Threshold value to convert predictions to binary. Defaults to 0.5.

    Returns:
        torch.Tensor:
            The average surface distance.
    """

    predictions_, targets_ = (predictions > threshold).detach().cpu().numpy(), targets.detach().cpu().numpy()

    predictions_, targets_ = handle_empty_array(predictions_), handle_empty_array(targets_)

    return asd(predictions_, targets_)
