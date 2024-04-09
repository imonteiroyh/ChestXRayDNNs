import torch


def dice(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the Dice score metric for a batch of predictions.

    Parameters:
        predictions (torch.Tensor):
            The predicted segmentation masks. Expected to be in one-hot format with the first dimension
            representing the batch size. The values should be binarized. Example shape: [batch_size,
            num_classes, height, width].
        targets (torch.Tensor):
            The ground truth segmentation masks. Expected to be in one-hot format with the first dimension
            representing the batch size. The values should be binarized. Example shape: [batch_size,
            num_classes, height, width].

    Returns:
        torch.Tensor:
            The Dice scores per batch and per class. The shape is [batch_size, num_classes].

    Raises:
        ValueError:
            If the shapes of `predictions` and `targets` are not the same.
    """

    targets_ = targets.float()
    predictions_ = predictions.float()

    if targets_.shape != predictions_.shape:
        raise ValueError(
            f"y_pred and y should have same shapes, got {predictions_.shape} and {targets_.shape}."
        )

    n_dims = len(predictions_.shape)
    reduce_axes = list(range(2, n_dims))
    intersection = torch.sum(targets_ * predictions_, dim=reduce_axes)

    targets_sum = torch.sum(targets_, reduce_axes)
    predictions_sum = torch.sum(predictions_, dim=reduce_axes)
    denominator = targets_sum + predictions_sum

    return torch.where(
        denominator > 0, (2.0 * intersection) / denominator, torch.tensor(1.0, device=targets_sum.device)
    ).mean()
