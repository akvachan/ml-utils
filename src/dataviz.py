#!/usr/bin/env python3

"""
Module Name: dataviz.py

Description:
    Utilities for data visualization using matplotlib.

Author: Arsenii Kvachan
MIT License, 2025
"""

import matplotlib
import matplotlib.pyplot as plt
from typing import Optional, Union


def plot(
    x: Union[list[float], list[list[float]]],
    y: Optional[Union[list[float], list[list[float]]]] = None,
    fmts: list[str] = ["-", "m--", "g-.", "r:"],
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (10, 6),
) -> None:
    """
    Method to visualize data via pyplot.
    """
    plt.figure(figsize=figsize)

    def is_vector(v):
        return isinstance(v[0], float)

    if y is None:
        if is_vector(x):
            plt.plot(x, fmts[0])
        else:
            for i, yvals in enumerate(x):
                plt.plot(yvals, fmts[i % len(fmts)])
    else:
        x_list = [x] if is_vector(x) else x
        y_list = [y] if is_vector(y) else y
        for i in range(len(y_list)):
            xi = x_list[i] if len(x_list) > 1 else x_list[0]
            plt.plot(xi, y_list[i], fmts[i % len(fmts)])

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def use_kitcat_backend() -> None:
    """
    Method to enable plot output in the terminal.
    """
    matplotlib.use("kitcat")
