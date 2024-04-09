import torch
from torchmetrics.functional import jaccard_index as jaccard_index_


def jaccard_index(predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Calculates the Jaccard Index metric for multilabel segmentation tasks.

    Parameters:
        predictions (torch.Tensor):
            The predicted tensor.
        targets (torch.Tensor):
            The target tensor.
        **kwargs:
            Additional keyword arguments to be passed to the underlying `jaccard_index_` function.

    Returns:
        torch.Tensor:
            The Jaccard Index value.
    """

    predictions_, targets_ = predictions.view(predictions.size(0), -1), targets.view(targets.size(0), -1)

    return jaccard_index_(predictions_, targets_, task="multilabel", num_labels=targets_.shape[1], **kwargs)
