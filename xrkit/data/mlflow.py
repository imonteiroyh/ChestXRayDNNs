import json
from typing import Dict, List, Tuple

import pytorch_lightning as L
import torch
import torch.utils

# mypy: disable-error-code="attr-defined, union-attr"


def save_results(
    predict_results: List[List[Tuple[float, float]]],
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    trainer: L.Trainer,
) -> None:
    """
    Save prediction results to MLflow experiment

    Args:
        predict_results (List[List[Tuple[float, float]]]): List of prediction results for each dataloader
        dataloaders (Dict[str, torch.utils.data.DataLoader]): Dictionary of dataloaders
        trainer (L.Trainer): PyTorch Lightning trainer
    """

    for dataloader in dataloaders:
        index = list(dataloaders.keys()).index(dataloader)

        outputs = [predict_results[index][batch][0] for batch in range(len(predict_results[index]))]
        targets = [predict_results[index][batch][1] for batch in range(len(predict_results[index]))]

        for filename, data in {
            f"{dataloader}_outputs.json": outputs,
            f"{dataloader}_targets.json": targets,
        }.items():

            dumped_data = json.dumps(
                [batch_results.tolist() for batch in data for batch_results in batch], indent=4
            )

            try:
                trainer.logger.experiment.log_text(
                    trainer.logger.run_id,
                    dumped_data,
                    filename,
                )

            except AttributeError:
                pass
