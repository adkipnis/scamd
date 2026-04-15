"""Visualization helpers for synthetic tabular datasets."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plotDataset(
    x: np.ndarray | pd.DataFrame,
    names: list[str] | None = None,
    color: str = 'green',
    alpha: float = 0.75,
    title: str = '',
    kde: bool = True,
):
    """Plot a pair grid with histograms, scatter plots, and optional KDEs."""
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'plotDataset requires pandas; install scamd plotting dependencies'
        ) from exc

    import seaborn as sns

    def _histplot(values, **kwargs):
        """Draw a detached histogram on a twin y-axis."""
        ax = plt.gca()
        ax2 = ax.twinx()
        sns.histplot(values, **kwargs, ax=ax2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax2.set_ylabel('')
        ax2.set_yticks([])
        ax2.set_yticklabels([])

    if isinstance(x, pd.DataFrame):
        data = x.select_dtypes(include=['number']).copy()
    else:
        arr = np.asarray(x)
        if arr.ndim != 2:
            raise ValueError('x must be a 2D array or DataFrame')
        mask = ~np.isnan(arr).any(axis=0)
        data = pd.DataFrame(arr[:, mask])

    n, d = data.shape
    if d == 0:
        raise ValueError('x must contain at least one numeric feature column')

    if names is not None:
        if len(names) != d:
            raise ValueError(
                'names length must match number of plotted columns'
            )
        labels = names
    elif isinstance(x, pd.DataFrame):
        labels = [str(c) for c in data.columns]
    else:
        labels = [rf'$x_{i + 1}$' for i in range(d)]

    g = sns.PairGrid(data, height=2.5)
    g.map_diag(
        _histplot,
        color=color,
        alpha=alpha,
        kde=False,
        fill=True,
        stat='density',
        common_norm=False,
    )

    alpha_point = min(0.85, max(0.08, 1.0 / np.log(max(n, 3))))
    g.map_upper(
        sns.scatterplot,
        color=color,
        alpha=alpha_point,
        s=22,
        edgecolor='k',
        lw=0,
    )

    if kde:
        g.map_lower(
            sns.kdeplot,
            color=color,
            alpha=alpha,
            fill=True,
            warn_singular=False,
        )
    else:
        g.map_lower(
            sns.scatterplot,
            color=color,
            alpha=alpha_point,
            s=22,
            edgecolor='k',
            lw=0,
        )

    for i in range(d):
        g.axes[i, 0].set_ylabel(labels[i], fontsize=12)
        g.axes[d - 1, i].set_xlabel(labels[i], fontsize=12)
        for j in range(d):
            g.axes[i, j].grid(alpha=0.35)
            g.axes[i, j].set_axisbelow(True)

    for i, ax in enumerate(g.axes[0, :]):
        xlabel = g.axes[-1, i].get_xlabel()
        g.fig.text(
            ax.get_position().x0 + ax.get_position().width / 2,
            ax.get_position().y1,
            xlabel,
            ha='center',
            va='bottom',
            fontsize=12,
        )
    for i in range(g.axes.shape[1]):
        g.axes[-1, i].set_xlabel('')

    if title:
        g.figure.suptitle(title, fontsize=16, y=1.01)
    g.tight_layout()
    return g
