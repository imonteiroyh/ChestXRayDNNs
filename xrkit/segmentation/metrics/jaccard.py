import torch


def jaccard_index(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Calculates the Jaccard Index metric for a batch of predictions.

    Parameters:
        predictions (torch.Tensor):
            The predicted tensor.
        targets (torch.Tensor):
            The target tensor.
        threshold (float):
            The threshold above which predictions are considered positive. Defaults to 0.5.

    Returns:
        torch.Tensor:
            The Jaccard Index value.
    """

    predictions_, targets_ = predictions.view(predictions.size(0), -1), targets.view(targets.size(0), -1)

    y_pred_positive = torch.round((predictions_ > threshold).int())
    y_pred_negative = 1 - y_pred_positive

    y_positive = torch.round(torch.clip(targets_, 0, 1))
    y_negative = 1 - y_positive

    TP = (y_positive * y_pred_positive).sum(dim=1)
    _ = (y_negative * y_pred_negative).sum(dim=1)

    FP = (y_negative * y_pred_positive).sum(dim=1)
    FN = (y_positive * y_pred_negative).sum(dim=1)

    return ((TP) / (TP + FP + FN)).mean()
