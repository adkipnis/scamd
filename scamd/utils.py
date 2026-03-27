"""Randomness and array utilities for dataset generation."""

import random
import numpy as np
import torch

# --- sampling
def setSeed(s: int) -> None:
    global _RNG
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
    """sample from log uniform in [low, high) + {add} and optionally floor to integer"""
    assert 0 < low, 'lower bound must be positive'
    assert low <= high, 'lower bound smaller than upper bound'
    log_low = np.log(low)
    log_high = np.log(high)
    out = rng.uniform(log_low, log_high, size)
    out = np.exp(out) + add
    if round:
        out = np.floor(out).astype(int)
    return out


# --- sanity checkers

def hasConstantColumns(x: torch.Tensor) -> torch.Tensor:
    #  numeric (n, d) -> bool (d,)
    first_row = x[0]
    col_is_constant = (x == first_row).all(dim=0)
    return col_is_constant

def sanityCheck(x: torch.Tensor) -> bool:
    if hasConstantColumns(x).any():
        return False
    if not torch.isfinite(x).any():
        return False
    return True


# --- standardize
def moments(
    x: np.ndarray,
    axis: int = 0,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis, keepdims=True)
    std = x.std(axis, keepdims=True)
    if exclude is not None:
        exclude = exclude.reshape(mean.shape)
        mean[exclude] = 0
        std[exclude] = 1
    return mean, std


def standardize(
    x: np.ndarray,
    axis: int = 0,
    exclude_binary: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    exclude = checkBinary(x, axis=axis) if exclude_binary else None
    mean, std = moments(x, axis, exclude=exclude)
    bad = (~np.isfinite(std)) | (std < eps)
    std = np.where(bad, 1.0, std)
    return (x - mean) / std
