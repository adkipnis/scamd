"""Visualize behavior of meta layers in ``scamd.meta``.

This mirrors the style of the other example scripts: construct a fixed input,
apply each meta component, and plot the resulting transforms.
"""

import matplotlib.pyplot as plt
import torch
from torch import nn

from scamd.meta import (
    RandomChoice,
    RandomScale,
    RandomScaleFactory,
    Standardizer,
)
from scamd.utils import setSeed


def plot_meta_layers() -> None:
    """Plot outputs of core meta layers on a shared 1D input grid."""
    # Standardizer computes stats over dim=0, so we keep dim=0 > 1.
    x = torch.linspace(-4, 4, 400).view(-1, 1)
    x_np = x[:, 0].numpy()

    # Instantiate layers once so random draws are visible and reproducible.
    standardizer = Standardizer()
    random_scale = RandomScale()
    scaled_tanh = RandomScaleFactory(nn.Tanh)()
    random_choice = RandomChoice([nn.Identity, nn.ReLU, nn.Tanh], n_choice=1)

    y_standardized = standardizer(x)
    y_scaled = random_scale(x)
    y_scaled_tanh = scaled_tanh(x)
    y_choice = random_choice(x)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = list(axes.flat)

    axes[0].plot(x_np, y_standardized[:, 0].numpy())
    axes[0].set_title('Standardizer')

    axes[1].plot(x_np, y_scaled[:, 0].numpy())
    axes[1].set_title('RandomScale')

    axes[2].plot(x_np, y_scaled_tanh[:, 0].detach().numpy())
    axes[2].set_title('RandomScaleFactory(nn.Tanh)')

    axes[3].plot(x_np, y_choice[:, 0].detach().numpy())
    axes[3].set_title('RandomChoice(ID, ReLU, Tanh)')

    for ax in axes:
        ax.grid(alpha=0.2)

    plt.suptitle('Meta layer transforms', size=16)
    fig.tight_layout()


if __name__ == '__main__':
    setSeed(3)
    plot_meta_layers()
