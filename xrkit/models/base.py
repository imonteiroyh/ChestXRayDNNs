from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from xrkit.base import CONFIG

# mypy: disable-error-code="attr-defined, arg-type"


class AutoEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        """
        Initializes an AutoEncoder module.

        Parameters:
            encoder (nn.Module):
                The encoder module.
            decoder (nn.Module):
                The decoder module.
        """

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the AutoEncoder.

        Parameters:
            inputs (torch.Tensor):
                The input tensor to be encoded and decoded.

        Returns:
            torch.Tensor:
                The decoded output tensor.
        """

        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)

        return decoded


class BaseModel:
    def setup_model(
        self,
        network: torch.nn.Module,
        criterion: torch.nn.Module,
        metrics: Iterable[Tuple[Callable, Dict[str, Any]]],
        activation_function: Callable,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        on_fit_start: Optional[Callable] = None,
        training_step: Optional[Callable] = None,
        validation_step: Optional[Callable] = None,
        test_step: Optional[Callable] = None,
        on_train_epoch_end: Optional[Callable] = None,
        on_validation_epoch_end: Optional[Callable] = None,
        on_test_epoch_end: Optional[Callable] = None,
        configure_optimizers: Optional[Callable] = None,
    ) -> None:
        """
        Method to configure the model with various components such as network architecture, loss function,
        metrics, etc.

        Args:
            network (torch.nn.Module):
                The neural network model.
            criterion (torch.nn.Module):
                The loss function.
            metrics (Iterable[Tuple[Callable, Dict[str, Any]]]):
                Iterable of tuples containing metric functions and their parameters.
            activation_function (Callable):
                The activation function to be applied to the network outputs.
            optimizer (torch.optim.Optimizer):
                The optimizer for updating model parameters.
            scheduler (Any):
                The learning rate scheduler.
            on_fit_start (Optional[Callable]):
                Callback function to be executed at the beginning of model fitting.
            training_step (Optional[Callable]):
                Custom training step function.
            validation_step (Optional[Callable]):
                Custom validation step function.
            test_step (Optional[Callable]):
                Custom test step function.
            on_train_epoch_end (Optional[Callable]):
                Callback function to be executed at the end of each training epoch.
            on_validation_epoch_end (Optional[Callable]):
                Callback function to be executed at the end of each validation epoch.
            on_test_epoch_end (Optional[Callable]):
                Callback function to be executed at the end of each testing epoch.
            configure_optimizers (Optional[Callable]):
                Function to configure the optimizers.
        """

        self.network = network
        self.criterion = criterion
        self.metrics = metrics
        self.activation_function = activation_function
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.training_step_outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.validation_step_outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.test_step_outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []

        self.on_fit_start = on_fit_start or self.on_fit_start_override
        self.training_step = training_step or self.training_step_override
        self.validation_step = validation_step or self.validation_step_override
        self.test_step = test_step or self.test_step_override
        self.on_train_epoch_end = on_train_epoch_end or self.on_train_epoch_end_override
        self.on_validation_epoch_end = on_validation_epoch_end or self.on_validation_epoch_end_override
        self.on_test_epoch_end = on_test_epoch_end or self.on_test_epoch_end_override
        self.configure_optimizers = configure_optimizers or self.configure_optimizers_override

    def on_fit_start_override(self) -> None:
        """Callback function executed at the beginning of model fitting."""

        try:
            self.logger.experiment.log_param(self.logger.run_id, "network", self.network.__class__.__name__)

            if hasattr(self.network, "encoder"):
                self.logger.experiment.log_param(
                    self.logger.run_id, "encoder", self.network.encoder.__class__.__name__
                )

            if hasattr(self.network, "decoder"):
                self.logger.experiment.log_param(
                    self.logger.run_id, "decoder", self.network.decoder.__class__.__name__
                )

            self.logger.experiment.log_param(
                self.logger.run_id, "criterion", self.criterion.__class__.__name__
            )
            self.logger.experiment.log_param(self.logger.run_id, "batch_size", CONFIG.base.batch_size)
            self.logger.experiment.log_param(
                self.logger.run_id, "optimizer", self.optimizer.__class__.__name__
            )
            self.logger.experiment.log_param(
                self.logger.run_id, "scheduler", self.scheduler.__class__.__name__
            )
            self.logger.experiment.log_param(
                self.logger.run_id, "activaction_function", self.activaction_function.__name__
            )

            metrics = ""
            for function_, parameters in self.metrics:
                metrics += f"{function_.__name__}:"
                for key, value in parameters.items():
                    metrics += f"\n\t{key}: {value}"
                metrics += "\n\n"
            self.logger.experiment.log_text(self.logger.run_id, metrics, "metrics.txt")

            self.logger.experiment.log_text(self.logger.run_id, str(self.network), "summary.txt")
        except AttributeError:
            pass

    def __log_metrics(self, outputs: List[torch.Tensor], targets: List[torch.Tensor], mode: str) -> None:
        """
        Helper method to log evaluation metrics.

        Args:
            outputs (List[torch.Tensor]):
                Model predictions.
            targets (List[torch.Tensor]):
                Ground truth labels.
            mode (str):
                Indicates the mode of evaluation (train/validation/test).
        """

        all_outputs, all_targets = torch.cat(outputs), torch.cat(targets)
        all_targets = all_targets.int()

        for metric_fn, kwargs in self.metrics:
            result = metric_fn(all_outputs, all_targets, **kwargs)
            self.log(f"{mode}_{metric_fn.__name__}", result, on_step=False, on_epoch=True, logger=True)

    def __shared_evaluation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shared evaluation step for training, validation, and testing.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]):
                Input data batch.
            batch_index (int):
                Index of the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Loss, model outputs, and ground truth targets.
        """

        inputs, targets = batch
        outputs = self.network(inputs)
        loss = self.criterion(outputs, targets)

        outputs = self.activation_function(outputs)

        return loss, outputs, targets

    def training_step_override(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int
    ) -> torch.Tensor:
        """
        Custom training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]):
                Input data batch.
            batch_index (int):
                Index of the batch.

        Returns:
            torch.Tensor:
                Loss value.
        """

        loss, outputs, targets = self.__shared_evaluation_step(batch, batch_index)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.training_step_outputs.append((outputs, targets))

        return loss

    def validation_step_override(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int
    ) -> torch.Tensor:
        """
        Custom validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]):
                Input data batch.
            batch_index (int):
                Index of the batch.

        Returns:
            torch.Tensor:
                Loss value.
        """

        loss, outputs, targets = self.__shared_evaluation_step(batch, batch_index)
        self.log("validation_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.validation_step_outputs.append((outputs, targets))

        return loss

    def test_step_override(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int) -> torch.Tensor:
        """
        Custom test step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]):
                Input data batch.
            batch_index (int):
                Index of the batch.

        Returns:
            torch.Tensor:
                Loss value.
        """

        loss, outputs, targets = self.__shared_evaluation_step(batch, batch_index)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.test_step_outputs.append((outputs, targets))

        return loss

    def on_train_epoch_end_override(self) -> None:
        """Callback function executed at the end of each training epoch."""

        outputs, targets = zip(*self.training_step_outputs)
        self.__log_metrics(outputs, targets, "train")
        self.training_step_outputs.clear()

    def on_validation_epoch_end_override(self) -> None:
        """Callback function executed at the end of each validation epoch."""

        outputs, targets = zip(*self.validation_step_outputs)
        self.__log_metrics(outputs, targets, "validation")
        self.validation_step_outputs.clear()

    def on_test_epoch_end_override(self) -> None:
        """Callback function executed at the end of each testing epoch."""

        outputs, targets = zip(*self.test_step_outputs)
        self.__log_metrics(outputs, targets, "test")
        self.test_step_outputs.clear()

    def configure_optimizers_override(self) -> Tuple[List[Any], List[Any]]:
        """
        Override this method to configure the optimizers.

        Returns:
            Tuple[List[Any], List[Any]]:
                Tuple containing optimizer and scheduler.
        """

        optimizer = self.optimizer
        scheduler = self.scheduler

        return [optimizer], [scheduler]
