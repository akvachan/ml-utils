#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module Name: dataviz.py

Description:
    Utilities for data visualization using matplotlib.

Author: Arsenii Kvachan
MIT License, 2025
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

# Typedefs
from typing import Optional, Union, List, Tuple, Sequence


def use_kitcat_backend() -> None:
    """
    Set matplotlib to use kitcat backend for terminal-based image rendering.
    """
    matplotlib.use("kitcat")


def set_figsize(figsize: Tuple[float, float] = (17.5, 17.5)) -> None:
    """
    Set the global default figure size for matplotlib plots.

    Parameters:
        figsize (tuple): A tuple of width and height in inches.
    """
    plt.rcParams['figure.figsize'] = figsize


def set_axes(
    axes: Axes,
    xlabel: Optional[str],
    ylabel: Optional[str],
    xlim: Optional[Tuple[float, float]],
    ylim: Optional[Tuple[float, float]],
    xscale: str,
    yscale: str,
    legend: Union[List[str], Tuple[str, ...], None]
) -> None:
    """
    Configure axes with labels, limits, scales, legend, and grid.

    Parameters:
        axes (Axes): A matplotlib Axes object to configure.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        xlim (tuple, optional): Limits for the x-axis.
        ylim (tuple, optional): Limits for the y-axis.
        xscale (str): Scale for the x-axis ('linear', 'log', etc.).
        yscale (str): Scale for the y-axis ('linear', 'log', etc.).
        legend (list or tuple, optional): Labels for the plot legend.
    """
    if xlabel:
        axes.set_xlabel(xlabel)
    if ylabel:
        axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    if xlim:
        axes.set_xlim(xlim)
    if ylim:
        axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(
    X: Union[Sequence[float], Sequence[Sequence[float]], np.ndarray],
    Y: Optional[Union[Sequence[float],
                      Sequence[Sequence[float]], np.ndarray]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: Union[List[str], Tuple[str, ...]] = [],
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xscale: str = 'linear',
    yscale: str = 'linear',
    fmts: Tuple[str, ...] = ('-', 'm--', 'g-.', 'r:'),
    figsize: Tuple[float, float] = (17.5, 17.5),
    axes: Optional[Axes] = None
) -> None:
    """
    Plot data using matplotlib with flexible support for multiple series.

    Parameters:
        X (list, ndarray): Input x-values. Can be a list of lists.
        Y (list, ndarray, optional): Input y-values.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        legend (list or tuple, optional): Labels for each plotted line.
        xlim (tuple, optional): Limits for the x-axis.
        ylim (tuple, optional): Limits for the y-axis.
        xscale (str): Scale type for the x-axis.
        yscale (str): Scale type for the y-axis.
        fmts (tuple): Format strings for plot styles.
        figsize (tuple): Size of the figure in inches.
        axes (Axes, optional): Matplotlib Axes to plot on.
    """
    def has_one_axis(X: Union[Sequence[float], np.ndarray]) -> bool:
        """
        Determine if input is a 1D structure.

        Parameters:
            X: Data to check.

        Returns:
            bool: True if X is 1D, False otherwise.
        """
        return (hasattr(X, "ndim") and X.ndim == 1 or
                isinstance(X, list) and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None:
        axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
