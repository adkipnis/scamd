"""Compare SCM-only and SCM+Posthoc dependence profiles across presets.

This demo mirrors the pool demo style by plotting multiple practical presets
and showing how post-hoc transforms reshape feature dependencies.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scamd.plotting import plot_dataset
from scamd.pool import getActivations
from scamd.presets import DATASET_PRESETS, PRESET_LABELS, get_pool_preset
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
    n_causes: int,
    n_layers: int,
    n_hidden: int,
    blockwise: bool,
    p_posthoc: float,
    cause_dist: str,
    fixed: bool,
    pool_kwargs: dict,
    **scm_extra,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one dataset before and after Posthoc transformation."""
    pool = getActivations(**pool_kwargs)
    activation = pool[int(getRng().integers(0, len(pool)))]

    config = {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_causes': n_causes,
        'n_layers': n_layers,
        'n_hidden': n_hidden,
        'blockwise': blockwise,
        'cause_dist': cause_dist,
        'fixed': fixed,
        'activation': activation,
        **scm_extra,
    }
    scm = SCM(**config)
    x = scm.sample()
    base = x.detach().cpu().numpy()
    posthoc = Posthoc(n_features=n_features, p_posthoc=p_posthoc)
    transformed = posthoc(x)
    return base, transformed


def plot_scm_presets() -> None:
    """Plot dependence spectra for three realistic SCM presets."""
    preset_names = ['balanced_realistic', 'smooth_stable', 'high_variability']
    presets = []
    for name in preset_names:
        cfg = DATASET_PRESETS[name]
        presets.append(
            (
                PRESET_LABELS[name],
                dict(
                    n_samples=1500,
                    n_features=24,
                    n_causes={
                        'balanced_realistic': 12,
                        'smooth_stable': 10,
                        'high_variability': 16,
                    }[name],
                    n_layers={
                        'balanced_realistic': 8,
                        'smooth_stable': 6,
                        'high_variability': 10,
                    }[name],
                    n_hidden={
                        'balanced_realistic': 64,
                        'smooth_stable': 56,
                        'high_variability': 96,
                    }[name],
                    blockwise={
                        'balanced_realistic': True,
                        'smooth_stable': False,
                        'high_variability': True,
                    }[name],
                    p_posthoc=cfg['p_posthoc'],
                    cause_dist=cfg['cause_dist'],
                    fixed=cfg['fixed'],
                    pool_kwargs=get_pool_preset(cfg['pool_preset']),
                    contiguous=(name == 'smooth_stable'),
                    sigma_e={
                        'balanced_realistic': 0.02,
                        'smooth_stable': 0.01,
                        'high_variability': 0.04,
                    }[name],
                    vary_sigma_e=(name != 'smooth_stable'),
                ),
            )
        )

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
    base_cfg = DATASET_PRESETS['balanced_realistic']
    cfg = dict(
        n_samples=1200,
        n_features=8,
        n_causes=12,
        n_layers=8,
        n_hidden=64,
        blockwise=True,
        p_posthoc=base_cfg['p_posthoc'],
        cause_dist=base_cfg['cause_dist'],
        fixed=base_cfg['fixed'],
        pool_kwargs=get_pool_preset(base_cfg['pool_preset']),
        contiguous=False,
        sigma_e=0.02,
        vary_sigma_e=True,
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
