#!/usr/bin/env python3

"""
Module: abstract.py

Description:
    Simplified abstract base classes for deep learning models, data handling, and training.
    Conforms to modern Python and PyTorch conventions, avoids inspect-based magic.

Author: Adapted by ChatGPT
License: MIT License, 2025
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class ProgressBoard(ABC):
    """
    Interface for a progress board that can visualize training/validation metrics.
    """

    @abstractmethod
    def draw(
        self,
        x: float,
        y: float,
        label: str,
    ) -> None:
        """
        Draw a point or update a metric on the board.

        Args:
            x: The x-coordinate (e.g., epoch or fraction of epoch).
            y: The metric value.
            label: A label for the metric (e.g., 'loss', 'accuracy').
        """
        ...


class Module(nn.Module, ABC):
    """
    Base class for neural network models with built-in training/validation step hooks.
    """

    def __init__(
        self,
        board: Optional[ProgressBoard] = None,
    ) -> None:
        super().__init__()
        self.board: Optional[ProgressBoard] = board
        self.current_step: int = 0
        self.current_epoch: int = 0

    @abstractmethod
    def loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss given model outputs and targets.
        """
        ...

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the network. Override in subclasses if needed.
        """
        return super().forward(x)

    def plot(
        self,
        x: float,
        y: float,
        label: str,
        train: bool = True,
    ) -> None:
        """
        Log a metric to the progress board.
        """
        if self.board is None:
            return
        phase = "train" if train else "val"
        self.board.draw(
            x=x,
            y=y,
            label=f"{phase}/{label}",
        )

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Logic for one training step.
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss_value = self.loss(
            outputs=outputs,
            targets=targets,
        )
        self.plot(
            x=self.current_step,
            y=loss_value.item(),
            label="loss",
            train=True,
        )
        self.current_step += 1
        return loss_value

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """
        Logic for one validation step.
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss_value = self.loss(
            outputs=outputs,
            targets=targets,
        )
        self.plot(
            x=float(self.current_epoch + 1),
            y=loss_value.item(),
            label="loss",
            train=False,
        )

    @abstractmethod
    def configure_optimizers(
        self,
    ) -> torch.optim.Optimizer:
        """
        Create and return optimizer(s) for training.
        """
        ...


class DataModule(ABC):
    """
    Base class for data modules that provide train/validation DataLoaders.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ) -> None:
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory

    @abstractmethod
    def get_dataset(
        self,
        train: bool,
    ) -> Dataset:
        """
        Returns a torch Dataset for train or validation.
        """
        ...

    def train_dataloader(
        self,
    ) -> DataLoader:
        """
        Return DataLoader for training set.
        """
        return DataLoader(
            dataset=self.get_dataset(train=True),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(
        self,
    ) -> DataLoader:
        """
        Return DataLoader for validation set.
        """
        return DataLoader(
            dataset=self.get_dataset(train=False),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class Trainer:
    """
    Simple trainer for orchestrating the training and validation loops.
    """

    def __init__(
        self,
        max_epochs: int,
        board: Optional[ProgressBoard],
        gradient_clip_val: float,
    ) -> None:
        self.max_epochs: int = max_epochs
        self.board: Optional[ProgressBoard] = board
        self.gradient_clip_val: float = gradient_clip_val

    def fit(
        self,
        model: Module,
        data_module: DataModule,
    ) -> None:
        """
        Fit the model using data from the provided DataModule.
        """
        optimizer = model.configure_optimizers()
        model.current_step = 0
        for epoch in range(self.max_epochs):
            model.current_epoch = epoch

            # Training phase
            model.train()
            for batch in data_module.train_dataloader():
                loss = model.training_step(batch)
                optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip_val > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=model.parameters(),
                        max_norm=self.gradient_clip_val,
                    )
                optimizer.step()

            # Validation phase
            model.eval()
            with torch.no_grad():
                for batch in data_module.val_dataloader():
                    model.validation_step(batch)
