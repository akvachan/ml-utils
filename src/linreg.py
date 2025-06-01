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
from typing import List


def solve_linreg_ana(
    X: List[List[int | float]],
    y: List[int | float],
) -> List[float]:
    """
    Solves linear regression analytically using the normal equation.
    Assumes the quadratic loss function (MSE).

    Parameters:
        X: 2D List of floats with shape (n_samples, n_features)
            Feature matrix, each row is a sample and each column is a feature.
        y: 1D List of floats with shape (n_samples,)
            Target values for each sample.

    Returns:
        1D List with shape (n_features + 1,)
        The optimal weight vector, where the last element is the bias term.

    Raises:
        ValueError: if input shapes are inconsistent,
                    or if the design matrix is not full rank
    """

    if not X or not y:
        return []

    if any(len(row) != len(X[0]) for row in X):
        raise ValueError("All rows of X must have the same number of features.")

    if len(X) != len(y):
        raise ValueError("int | float of samples in X and y must match.")

    X_np = np.array(X, dtype=np.float32)
    y_np = np.array(y, dtype=np.float32).reshape(-1, 1)

    X_with_bias = np.hstack([X_np, np.ones((X_np.shape[0], 1))])

    rank = np.linalg.matrix_rank(X_with_bias)
    if rank < min(X_with_bias.shape[0], X_with_bias.shape[1]):
        raise ValueError("Some features are linearly dependent. Cannot solve analytically.")

    w = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y_np
    return w.flatten().tolist()


def solve_linreg_sgd(
    X: List[List[int | float]],
    y: List[int | float],
    epochs: int = 100,
    batch_size: int = 5,
    learning_rate: float = 0.001,
):
    """
    Simple linear regression via mini-batch SGD.
    Returns a parameter vector of length (n_features + 1) with bias.

    Parameters:
        X: 2D List of floats with shape (n_samples, n_features)
            Feature matrix, each row is a sample and each column is a feature.
        y: 1D List of floats with shape (n_samples,)
            Target values for each sample.
        epochs: number of iterations.
        batch_size: number of samples in one batch.
        learning_rate: learning rate coefficient, size of the step.

    Returns:
        1D List with shape (n_features + 1,)
        The optimal weight vector, where the last element is the bias term.

    Raises:
        ValueError: if input shapes are inconsistent.
    """
    X_np = np.array(X, dtype=float)
    y_np = np.array(y, dtype=float)

    if len(X_np) != len(y_np):
        raise ValueError("Number of rows in X must match length of y.")

    n_samples, n_features = X_np.shape
    w = np.random.uniform(1, -1, n_features)
    b = 0.0

    for _ in range(epochs):
        for start in range(0, n_samples, batch_size):
            Xb = X_np[start : start + batch_size]
            yb = y_np[start : start + batch_size]

            errs = Xb.dot(w) + b - yb
            grad_w = (Xb.T.dot(errs)) / len(yb)
            grad_b = errs.mean()

            w -= learning_rate * grad_w
            b -= learning_rate * grad_b

    return list(w) + [b]
