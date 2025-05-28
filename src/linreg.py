#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module Name: linreg.py

Description:
    Linear regression module for various implementations of linear regression.

Author: Arsenii Kvachan
MIT License, 2025
"""

import numpy as np

# Typedefs
from typing import List, TypeVar

Number = TypeVar("Number", int, float)


def solve_linreg(
    X: List[List[Number]],
    y: List[Number],
) -> List[Number]:
    """
    Solves linear regression analytically using the normal equation.
    Assumes the quadratic loss function (MSE).

    Parameters:
        X: 2D List of floats with shape (n_samples, n_features)
            Feature matrix where each row is a data sample and each column is a feature.
        y: 1D List of floats with shape (n_samples,)
            Target values for each sample.

    Returns:
        1D List with shape (n_features + 1,)
        The optimal weight vector, where the last element is the bias term.

    Raises:
        ValueError: if input shapes are inconsistent, or if the design matrix is not full rank
                    (i.e., features are linearly dependent and cannot be solved analytically).
    """
    if not X or not y:
        return []

    if any(len(row) != len(X[0]) for row in X):
        raise ValueError("All rows of X must have the same number of features.")

    if len(X) != len(y):
        raise ValueError("Number of samples in X and y must match.")

    X_np = np.array(X, dtype=np.float32)
    y_np = np.array(y, dtype=np.float32).reshape(-1, 1)

    X_with_bias = np.hstack([X_np, np.ones((X_np.shape[0], 1))])

    rank = np.linalg.matrix_rank(X_with_bias)
    if rank < min(X_with_bias.shape[0], X_with_bias.shape[1]):
        raise ValueError(
            "Some features are linearly dependent. Cannot solve analytically."
        )

    w = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y_np
    return w.flatten().tolist()
