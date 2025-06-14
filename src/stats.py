#!/usr/bin/env python3

"""
Module Name: stats.py

Description:
    Statistics module.

Author: Arsenii Kvachan
MIT License, 2025
"""

import numpy as np


def normal(
    x: float | int,
    mu: float | int,
    sigma: float | int,
):
    p = 1 / np.sqrt(2 * np.pi * sigma**2)
    return p * np.exp(-0.5 * (x - mu) ** 2 / sigma**2)
