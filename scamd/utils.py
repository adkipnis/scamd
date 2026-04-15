"""Randomness and array utilities for dataset generation."""

import random

import numpy as np
import torch


def setSeed(s: int) -> None:
    """Seed Python, NumPy (legacy), and PyTorch global random states."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def logUniform(
    rng: np.random.Generator,
    low: float,
    high: float,
    size: int | tuple[int, ...] | None = None,
    add: float = 0.0,
    round: bool = False,
) -> np.ndarray | np.floating | np.integer:
    """Sample from log-uniform in [low, high) + add, optionally rounded to int."""
    assert 0 < low, 'lower bound must be positive'
    assert low <= high, 'lower bound must not exceed upper bound'
    log_low = np.log(low)
    log_high = np.log(high)
    out = rng.uniform(log_low, log_high, size)
    out = np.exp(out) + add
    if round:
        out = np.floor(out).astype(int)
    return out


def hasConstantColumns(x: torch.Tensor) -> torch.Tensor:
    """Return a bool mask of shape (d,) that is True for constant columns."""
    first_row = x[0]
    return (x == first_row).all(dim=0)


def sanityCheck(x: torch.Tensor) -> bool:
    """Return False if x has any constant or non-finite columns."""
    if hasConstantColumns(x).any():
        return False
    if not torch.isfinite(x).any():
        return False
    return True
