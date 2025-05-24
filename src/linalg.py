#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module Name: linalg.py

Description:
    Linear algebra module for matrix multiplication, transposition, reshape,
    eigenvalues, covariance, transformations and much more!

Author: Arsenii Kvachan
MIT License, 2025
"""

# Typedefs
from typing import List, TypeVar, Optional
Number = TypeVar("Number", int, float)


def dot_mat_vec(
    mat: List[List[Number]],
    vec: List[Number],
) -> List[Number]:
    """
    Simple matrix-vector dot-product without any optimizations
    on pure Python data types.

    Parameters:
        mat: 2D List with shape (n, m)
        vec: 1D List with shape (m,)

    Returns:
        1D List with shape (n,)

    Raises:
        ValueError: if column count does not match the vector length
    """

    if not mat:
        return []

    if any(len(row) != len(vec) for row in mat):
        raise ValueError("All rows must match vector length.")

    return [sum(a * b for a, b in zip(row, vec)) for row in mat]


def dot_vec_vec(
    vecA: List[Number],
    vecB: List[Number],
) -> Optional[Number]:
    """
    Simple matrix-vector dot-product without any optimizations
    on pure Python data types.

    Parameters:
        vecA: 1D List
        vecB: 1D List

    Returns:
        Scalar value or None

    Raises:
        ValueError: if lengths of vector do not match
    """

    if not vecA or not vecB:
        return None

    if len(vecA) != len(vecB):
        raise ValueError("Vectors should have same length.")

    return sum(vecA[i]*vecB[i] for i in range(len(vecA)))


def tpose_mat(
    mat: List[List[Number]],
) -> List[List[Number]]:
    """
    Transpose a 2D matrix.

    Parameters:
        mat: 2D List of shape (n, m)

    Returns:
        A new 2D List of shape (m, n)

    Raises:
        ValueError: if rows have different length
    """

    if not mat:
        return []

    if any(len(row) != len(mat[0]) for row in mat):
        raise ValueError("All rows must have the same length.")

    return [list(col) for col in zip(*mat)]


def rshape_mat(
    mat: List[List[Number]],
    new_shape: tuple[int, int]
) -> List[List[Number]]:
    """
    Reshape a 2D matrix using numpy.

    Parameters:
        mat: 2D List of shape (n, m)
        new_shape: Tuple with the dimensions of new shape

    Returns:
        A new 2D List of shape (m, n)
    """
    import numpy as np

    if not mat:
        return []

    return np.reshape(np.array(mat), shape=new_shape).tolist()


def mean_mat(
    mat: List[List[Number]],
    mode: str
) -> List[float]:
    """
    Calculate the mean of a 2D matrix by row or by column,
    based on a given mode.

    Parameters:
        mat: 2D List of shape
        mode: Specific mode of calculation, can be 'row' or 'column'

    Returns:
        List of mean values

    Raises:
        TypeError: if matrix is not a 2D list
        ValueError: if mode is invalid
    """
    if not mat:
        return []

    if not all(hasattr(row, "__iter__") for row in mat):
        raise TypeError("'mat' must be a 2D sequence of numbers.")

    match mode:
        case "row":
            return [sum(row)/len(row) for row in mat]
        case "column":
            return [sum(column)/len(column) for column in zip(*mat)]
        case _:
            raise ValueError(
                f"Unknown mode '{mode}'. Valid values are 'row' and 'column'.")


def mult_mat_mat(
    matA: List[List[Number]],
    matB: List[List[Number]],
) -> List[List[Number]]:
    """
    The elementwise product of two matrices.

    Parameters:
        matA: 2D List with shape (n, m)
        matB: 2D List with shape (n, m)

    Returns:
        2D List with shape (n, m)

    Raises:
        ValueError: if dimensions are not compatible
    """

    if not matA or not matB:
        return [[]]

    if any(len(rowA) != len(rowB) for rowA in matA for rowB in matB) \
            or len(matA) != len(matB):
        raise ValueError(
            "Rows and columns of matrix A and B should have same length.")

    for i in range(len(matA)):
        for j in range(len(matA[i])):
            matA[i][j] *= matB[i][j]

    return matA


def mult_mat_mat(
    matA: List[List[Number]],
    matB: List[List[Number]],
) -> List[List[Number]]:
    """
    Matrix–matrix multiplication.

    Parameters:
        matA: 2D List with shape (n, m)
        matB: 2D List with shape (n, m)

    Returns:
        2D List with shape (n, m)

    Raises:
        ValueError: if dimensions are not compatible
    """
    pass
