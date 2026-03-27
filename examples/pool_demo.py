"""Visualize activation pools produced by `getActivations` presets.

This demo compares a few pool configurations to show how the activation
distribution changes with GP/random-choice settings.
"""

import matplotlib.pyplot as plt
import torch

from scamd.pool import getActivations
from scamd.presets import POOL_PRESETS, PRESET_LABELS
from scamd.utils import getRng, setSeed


def _sample_pool_curves(
    pool: list, x: torch.Tensor, n_curves: int
) -> list[torch.Tensor]:
    """Instantiate random activation callables from a pool and evaluate on x."""
    idx = getRng().integers(0, len(pool), size=n_curves)
    curves = []
    for i in idx:
        layer = pool[int(i)]()
        y = layer(x).detach().float()
        curves.append(y[:, 0])
    return curves


def plot_activation_pools() -> None:
    """Plot sampled activation curves for three practical pool presets."""
    x = torch.linspace(-5, 5, 400).view(-1, 1)
    x_np = x[:, 0].numpy()

    preset_names = ['balanced_realistic', 'smooth_stable', 'high_variability']
    presets = [
        (PRESET_LABELS[name], POOL_PRESETS[name]) for name in preset_names
    ]

    fig, axes = plt.subplots(len(presets), 1, figsize=(10, 11), sharex=True)
    axes = list(getattr(axes, 'flat', [axes]))

    for ax, (title, kwargs) in zip(axes, presets):
        pool = getActivations(**kwargs)
        curves = _sample_pool_curves(pool, x, n_curves=12)
        for y in curves:
            ax.plot(x_np, y.numpy(), alpha=0.65, linewidth=1.1)
        ax.set_title(f'{title}  (pool size={len(pool)})', size=11)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel('x')
    plt.suptitle('Activation pool presets from getActivations', size=16)
    fig.tight_layout()


if __name__ == '__main__':
    setSeed(7)
    plot_activation_pools()
