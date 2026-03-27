"""Compare SCM-only and SCM+Posthoc dependence profiles across presets.

This demo mirrors the pool demo style by plotting multiple practical presets
and showing how post-hoc transforms reshape feature dependencies.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scamd.plotting import plot_dataset
from scamd.pool import getActivations
from scamd.scm import Posthoc, SCM
from scamd.utils import getRng, setSeed


def _offdiag_abs_corr(x: np.ndarray) -> np.ndarray:
    """Return absolute off-diagonal feature correlations."""
    corr = np.corrcoef(x, rowvar=False)
    upper = np.triu_indices(corr.shape[0], k=1)
    return np.abs(corr[upper])


def _sample_pipeline(
    n_samples: int,
    n_features: int,
    p_posthoc: float,
    scm_kwargs: dict,
    pool_kwargs: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one dataset before and after Posthoc transformation."""
    pool = getActivations(**pool_kwargs)
    activation = pool[int(getRng().integers(0, len(pool)))]

    config = {
        'n_samples': n_samples,
        'n_features': n_features,
        'activation': activation,
        **scm_kwargs,
    }
    scm = SCM(**config)
    x = scm.sample()
    base = x.detach().cpu().numpy()
    posthoc = Posthoc(n_features=n_features, p_posthoc=p_posthoc)
    transformed = posthoc(x)
    return base, transformed


def plot_scm_presets() -> None:
    """Plot dependence spectra for three realistic SCM presets."""
    presets = [
        (
            'Balanced Realistic',
            dict(
                n_samples=1500,
                n_features=24,
                p_posthoc=0.25,
                scm_kwargs=dict(
                    n_causes=12,
                    cause_dist='mixed',
                    fixed=False,
                    n_layers=8,
                    n_hidden=64,
                    contiguous=False,
                    blockwise=True,
                    sigma_e=0.02,
                    vary_sigma_e=True,
                ),
                pool_kwargs=dict(
                    n_gp=12,
                    n_random_choice=8,
                    random_scale=True,
                    gp_type_probs=(0.35, 0.25, 0.40),
                    n_choice=2,
                    allow_nested_random_choice=False,
                ),
            ),
        ),
        (
            'Smooth + Stable',
            dict(
                n_samples=1500,
                n_features=24,
                p_posthoc=0.15,
                scm_kwargs=dict(
                    n_causes=10,
                    cause_dist='normal',
                    fixed=True,
                    n_layers=6,
                    n_hidden=56,
                    contiguous=True,
                    blockwise=False,
                    sigma_e=0.01,
                    vary_sigma_e=False,
                ),
                pool_kwargs=dict(
                    n_gp=8,
                    n_random_choice=4,
                    random_scale=True,
                    gp_types=('se', 'matern'),
                    gp_type_probs=(0.7, 0.3),
                    n_choice=1,
                    allow_nested_random_choice=False,
                ),
            ),
        ),
        (
            'High Variability',
            dict(
                n_samples=1500,
                n_features=24,
                p_posthoc=0.45,
                scm_kwargs=dict(
                    n_causes=16,
                    cause_dist='mixed',
                    fixed=False,
                    n_layers=10,
                    n_hidden=96,
                    contiguous=False,
                    blockwise=True,
                    sigma_e=0.04,
                    vary_sigma_e=True,
                ),
                pool_kwargs=dict(
                    n_gp=20,
                    n_random_choice=12,
                    random_scale=True,
                    gp_type_probs=(0.2, 0.25, 0.55),
                    n_choice=3,
                    allow_nested_random_choice=False,
                ),
            ),
        ),
    ]

    fig, axes = plt.subplots(len(presets), 1, figsize=(10, 11), sharex=True)
    axes = list(getattr(axes, 'flat', [axes]))

    for ax, (title, cfg) in zip(axes, presets):
        base, transformed = _sample_pipeline(**cfg)
        base_dep = np.sort(_offdiag_abs_corr(base))
        post_dep = np.sort(_offdiag_abs_corr(transformed))
        quantile = np.linspace(0.0, 1.0, base_dep.size)

        ax.plot(quantile, base_dep, label='SCM only', linewidth=1.5)
        ax.plot(quantile, post_dep, label='SCM + Posthoc', linewidth=1.5)
        ax.set_title(
            (
                f'{title}  (mean |corr|: {base_dep.mean():.3f} -> {post_dep.mean():.3f})'
            ),
            size=11,
        )
        ax.grid(alpha=0.2)
        ax.legend(loc='upper left')

    axes[-1].set_xlabel('Correlation quantile')
    plt.suptitle('Dependence profile by SCM preset', size=16)
    fig.tight_layout()


def plot_pairgrid_example() -> None:
    """Show pair-grid structure for one realistic SCM preset."""
    cfg = dict(
        n_samples=1200,
        n_features=8,
        p_posthoc=0.3,
        scm_kwargs=dict(
            n_causes=12,
            cause_dist='mixed',
            fixed=False,
            n_layers=8,
            n_hidden=64,
            contiguous=False,
            blockwise=True,
            sigma_e=0.02,
            vary_sigma_e=True,
        ),
        pool_kwargs=dict(
            n_gp=12,
            n_random_choice=8,
            random_scale=True,
            gp_type_probs=(0.35, 0.25, 0.40),
            n_choice=2,
            allow_nested_random_choice=False,
        ),
    )
    base, transformed = _sample_pipeline(**cfg)
    cols = [f'x{i + 1}' for i in range(base.shape[1])]
    base_df = pd.DataFrame(base, columns=cols)
    post_df = pd.DataFrame(transformed, columns=cols)

    plot_dataset(
        base_df,
        color='#2f6f95',
        title='SCM only (Balanced Realistic)',
        kde=True,
    )
    plot_dataset(
        post_df,
        color='#3b7f4a',
        title='SCM + Posthoc (Balanced Realistic)',
        kde=True,
    )


if __name__ == '__main__':
    setSeed(7)
    plot_scm_presets()
    plot_pairgrid_example()
    plt.show()
