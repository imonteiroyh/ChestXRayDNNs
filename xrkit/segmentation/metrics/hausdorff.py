import torch
from scipy.spatial.distance import directed_hausdorff


def balanced_average_hausdorff_distance(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculates the balanced average Hausdorff distance between predictions and targets.

    Parameters:
        predictions (torch.Tensor):
            The predicted tensor.
        targets (torch.Tensor):
            The target tensor.

    Returns:
        torch.Tensor:
            The balanced average Hausdorff distance.
    """

    batch_size, channels, *_ = targets.shape
    values = torch.zeros((batch_size, channels))

    for batch in range(batch_size):
        for channel in range(channels):
            targets_, predictions_ = (
                targets[batch][channel].detach().cpu(),
                predictions[batch][channel].detach().cpu(),
            )
            values[batch, channel] = (
                directed_hausdorff(targets_, predictions_)[0]
                + directed_hausdorff(predictions_, targets_)[0]
                * predictions_.view(1, -1).size(1)
                / targets_.view(1, -1).size(1)
            ) / 2

    return values.mean()
