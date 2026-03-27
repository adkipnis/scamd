"""Visualize baseline and sampled activation functions"""

import matplotlib.pyplot as plt
import torch
from torch import nn

from scamd.basic import basic_activations
from scamd.gp import GP
from scamd.utils import setSeed


def gridplot(
    activations: list[nn.Module],
    x: torch.Tensor,
    nrow: int,
    ncol: int,
    figsize: tuple[int, int],
    title: str = '',
    act_kwargs: dict | None = None,
) -> None:
    kwargs = act_kwargs or {}
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = list(getattr(axes, 'flat', [axes]))
    for i, act in enumerate(activations):
        layer = act(**kwargs)
        y = layer(x)
        axes[i].plot(x, y)
        axes[i].set_title(str(layer).split('(')[0], size=10)
    for ax in axes[len(activations) :]:
        ax.set_visible(False)
    plt.suptitle(title, size=18)
    fig.tight_layout()


if __name__ == '__main__':
    setSeed(42)
    grid = torch.arange(start=-10, end=10, step=20 / 256)

    # --- Basic activation functions
    gridplot(
        basic_activations,
        x=grid,
        ncol=5,
        nrow=5,
        figsize=(8, 8),
        title='Fixed Activations',
    )

    # --- Gaussian Process activation functions
    # Squared Exponential kernel (smooth)
    gridplot(
        [GP] * 25,  # type: ignore
        x=grid,
        ncol=5,
        nrow=5,
        figsize=(8, 8),
        act_kwargs={'gp_type': 'se'},
        title='Sampled Activations',
    )

    # Matern kernel (spiky)
    gridplot(
        [GP] * 25,  # type: ignore
        x=grid,
        ncol=5,
        nrow=5,
        figsize=(8, 8),
        act_kwargs={'gp_type': 'matern'},
        title='Sampled Activations',
    )

    # Fractional kernel (middle ground)
    gridplot(
        [GP] * 25,  # type: ignore
        x=grid,
        ncol=5,
        nrow=5,
        figsize=(8, 8),
        act_kwargs={'gp_type': 'fractional'},
        title='Sampled Activations',
    )
