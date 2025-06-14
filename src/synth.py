#!/usr/bin/env python3

"""
Module Name: synth.py

Description:
    Everything related to synthetic data generation.

Author: Adapted by ChatGPT
License: MIT License, 2025
"""

import random
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from abstract import DataModule


class SyntheticRegressionData(DataModule):
    """
    Synthetic dataset for linear regression.
    """

    def __init__(
        self,
        w: Tensor,
        b: float,
        noise: float = 0.01,
        num_train: int = 1000,
        num_val: int = 1000,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        total_samples: int = num_train + num_val
        features: int = w.numel()

        X: Tensor = torch.randn(total_samples, features)
        noise_tensor: Tensor = torch.randn(total_samples, 1) * noise
        y: Tensor = X @ w.reshape(-1, 1) + b + noise_tensor

        self.w: Tensor = w.clone()
        self.b: float = b
        self.noise: float = noise
        self.num_train: int = num_train
        self.num_val: int = num_val
        self.X: Tensor = X
        self.y: Tensor = y

    def get_dataset(
        self,
        train: bool,
    ) -> TensorDataset:
        """
        Return a TensorDataset for training or validation.

        Args:
            train: Whether to return the training split.
        """
        start: int = 0 if train else self.num_train
        end: int = self.num_train + self.num_val
        X_split: Tensor = self.X[start:end]
        y_split: Tensor = self.y[start:end]
        return TensorDataset(X_split, y_split)


if __name__ == "__main__":
    w_tensor: Tensor = torch.tensor([2.0, -3.4])
    b_value: float = 4.2
    data: SyntheticRegressionData = SyntheticRegressionData(
        w=w_tensor,
        b=b_value,
        seed=42,
    )
    first_feature: Tensor = data.X[0]
    first_label: Tensor = data.y[0]
    print(
        "features:",
        first_feature,
        "\nlabel:",
        first_label,
    )
