"""Visualize baseline and sampled activation functions.

This script plots simple activations, GP variants, and random activation mixtures.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.activations import getActivations
from src.basic import basic_activations
from src.gp import GP
from src.utils import setSeed


def gridplot(
    activations: list[nn.Module],
    x: torch.Tensor,
    nrow: int,
    ncol: int,
    figsize: tuple[int, int],
    act_kwargs: dict | None = None,
) -> None:
    kwargs = act_kwargs or {}
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = axes.flat
    for i, act in enumerate(activations):
        layer = act(**kwargs)
        y = layer(x)
        try:
            axes[i].plot(x, y)
            axes[i].set_title(str(layer).split('(')[0], size=10)
        except Exception:
            continue
    fig.tight_layout()


if __name__ == '__main__':
    setSeed(42)
    x = torch.arange(start=-10, end=10, step=20 / 256)

    gridplot(basic_activations, x, 7, 4, (7, 10))
    gridplot([GP] * 25, x, 5, 5, (8, 8), {'gp_type': 'SE'})
    gridplot([GP] * 25, x, 5, 5, (8, 8), {'gp_type': 'Matern'})
    gridplot([GP] * 25, x, 5, 5, (8, 8), {'gp_type': 'Fract'})

    sampled = getActivations()
    gridplot(sampled[:25], x, 5, 5, (8, 8))
