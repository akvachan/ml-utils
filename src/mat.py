#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module Name: mat.py

Description:
    Matrix multiplication, transposition, reshape,
    eigenvalues, covariance, transformations and much more!

Author: Arsenii Kvachan
MIT License, 2024
"""

from typing import List, TypeVar

# Typedefs
Number = TypeVar("Number", int, float)


def dot_mat_vec(
    mat: List[Number],
    vec: List[Number],
) -> List[Number]:
    """
    Simple matrix-vector dot-product without any optimizations
    on pure Python data types.

    Time: O(|mat|*|vec|)
    Space: O(|mat|)

    Parameters:
        mat: 2D List with shape (n, m)
        vec: 1D List with shape (m,)

    Returns:
        1D List with shape (n,)
    """

    if not mat:
        return []

    if any(len(row) != len(vec) for row in mat):
        raise ValueError("All rows must match vector length.")

    return [sum(a * b for a, b in zip(row, vec)) for row in mat]


def tpose_mat(
    mat: List[List[Number]],
) -> List[List[Number]]:
    """
    Transpose a 2D matrix.

    Time: O(|mat|*|mat|)
    Space: O(|mat|)

    Parameters:
        mat: 2D List of shape (n, m)

    Returns:
        A new 2D List of shape (m, n).
    """

    if not mat:
        return []

    if any(len(row) != len(mat[0]) for row in mat):
        raise ValueError("All rows must have the same length.")

    return [list(col) for col in zip(*mat)]
