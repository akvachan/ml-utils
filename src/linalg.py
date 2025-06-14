#!/usr/bin/env python3

"""
Module Name: linalg.py

Description:
    Linear algebra module for matrix multiplication, transposition, reshape,
    eigenvalues, covariance, transformations and much more!

Author: Arsenii Kvachan
MIT License, 2025
"""


def dot_mat_vec(
    mat: list[list[int | float]],
    vec: list[int | float],
) -> list[int | float]:
    """
    Simple matrix-vector dot-product without any optimizations
    on pure Python data types.

    Raises:
        ValueError: if column count does not match the vector length
    """

    if not mat:
        return []

    if any(len(row) != len(vec) for row in mat):
        raise ValueError("All rows must match vector length.")

    return [sum(a * b for a, b in zip(row, vec, strict=False)) for row in mat]


def dot_vec_vec(
    vecA: list[int | float],
    vecB: list[int | float],
) -> int | float | None:
    """
    Simple matrix-vector dot-product without any optimizations
    on pure Python data types.

    Raises:
        ValueError: if lengths of vector do not match
    """

    if not vecA or not vecB:
        return None

    if len(vecA) != len(vecB):
        raise ValueError("Vectors should have same length.")

    return sum(vecA[i] * vecB[i] for i in range(len(vecA)))


def tpose_mat(
    mat: list[list[int | float]],
) -> list[list[int | float]]:
    """
    Transpose a 2D matrix.

    Raises:
        ValueError: if rows have different length
    """

    if not mat:
        return []

    if any(len(row) != len(mat[0]) for row in mat):
        raise ValueError("All rows must have the same length.")

    return [list(col) for col in zip(*mat, strict=False)]


def rshape_mat(
    mat: list[list[int | float]],
    new_shape: tuple[int, int],
) -> list[list[int | float]]:
    """
    Reshape a 2D matrix using numpy.
    """
    import numpy as np

    if not mat:
        return []

    return np.reshape(np.array(mat), shape=new_shape).tolist()


def mean_mat(
    mat: list[list[int | float]],
    mode: str,
) -> list[float]:
    """
    Calculate the mean of a 2D matrix by row or by column,
    based on a given mode.

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
            return [sum(row) / len(row) for row in mat]
        case "column":
            return [sum(column) / len(column) for column in zip(*mat, strict=False)]
        case _:
            raise ValueError(
                f"Unknown mode '{mode}'. Valid values are 'row' and 'column'."
            )


def mult_mat_mat(
    matA: list[list[int | float]],
    matB: list[list[int | float]],
) -> list[list[int | float]]:
    """
    The elementwise product of two matrices.

    Raises:
        ValueError: if dimensions are not compatible
    """

    if not matA or not matB:
        return [[]]

    if any(len(rowA) != len(rowB) for rowA in matA for rowB in matB) or len(
        matA
    ) != len(matB):
        raise ValueError("Rows and columns of matrix A and B should have same length.")

    for i in range(len(matA)):
        for j in range(len(matA[i])):
            matA[i][j] *= matB[i][j]

    return matA
