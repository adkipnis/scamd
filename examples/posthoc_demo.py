"""Visualize each post-hoc transform and its correlation with input x."""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from scamd.posthoc import POSTHOC_LAYERS
from scamd.utils import setSeed


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation on flattened arrays."""
    x = x.ravel()
    y = y.ravel()
    if x.std() == 0 or y.std() == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def plot_posthoc_grid() -> None:
    """Plot every available post-hoc transform against the same input grid."""
    n_plots = len(POSTHOC_LAYERS)
    ncol = 3
    nrow = math.ceil(n_plots / ncol)

    x = torch.linspace(-4, 4, 400).view(1, -1, 1)
    x_np = x[0, :, 0].numpy()

    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 8))
    axes = list(getattr(axes, 'flat', [axes]))

    for i, layer_cls in enumerate(POSTHOC_LAYERS):
        layer = layer_cls(n_in=1, n_out=1)
        y = layer(x).float()
        y_np = y[0, :, 0].detach().numpy()
        corr = pearson_corr(x_np, y_np)

        axes[i].scatter(x_np, y_np, s=8, alpha=0.5)
        axes[i].set_title(layer_cls.__name__, size=11)
        axes[i].text(
            0.04,
            0.95,
            f'corr={corr:.3f}',
            transform=axes[i].transAxes,
            va='top',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8},
        )

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    plt.suptitle('Post-hoc transforms vs input x', size=18)
    fig.tight_layout()


if __name__ == '__main__':
    setSeed(0)
    plot_posthoc_grid()
