import random
import numpy as np
import torch

# --- sampling
def setSeed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def logUniform(
    rng: np.random.Generator,
    low: float,
    high: float,
    size: int | tuple[int, ...],
    add: float = 0.0,
    round: bool = False,
) -> np.ndarray:
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


# --- checkers
def checkBinary(x: np.ndarray, axis: int = 0) -> np.ndarray:
    binary = (x == 0) + (x == 1)
    return np.all(binary, axis=axis)


def checkContinuous(x: np.ndarray, axis: int = 0, tol: float = 1e-12) -> np.ndarray:
    diffs = np.abs(x - x.round())
    too_large = diffs > tol
    return np.all(too_large, axis=axis)


def checkConstant(x: np.ndarray, axis: int = 0) -> np.ndarray:
    same = x[0, :] == x
    return np.all(same, axis=axis)


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
    x: np.ndarray, axis: int = 0, exclude_binary: bool = True, eps: float = 1e-6
) -> np.ndarray:
    exclude = checkBinary(x, axis=axis) if exclude_binary else None
    mean, std = moments(x, axis, exclude=exclude)
    bad = (~np.isfinite(std)) | (std < eps)
    std = np.where(bad, 1.0, std)
    return (x - mean) / std


