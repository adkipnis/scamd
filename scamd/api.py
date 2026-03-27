"""Public API for synthetic dataset generation."""

from typing import Any, Callable

import numpy as np

from .pool import getActivations
from .presets import get_dataset_preset, get_pool_preset
from .scm import Posthoc, SCM
from .utils import getRng


def generate_dataset(
    *,
    n_samples: int,
    n_features: int,
    n_causes: int,
    n_layers: int,
    n_hidden: int,
    blockwise: bool,
    preset: str = 'balanced_realistic',
    activation: Callable | None = None,
    p_posthoc: float | None = None,
    cause_dist: str | None = None,
    fixed: bool | None = None,
    **config: Any,
) -> np.ndarray:
    """Generate one standardized X-only synthetic dataset."""
    scm_config: dict[str, Any] = {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_causes': n_causes,
        'n_layers': n_layers,
        'n_hidden': n_hidden,
        'blockwise': blockwise,
    }
    posthoc_config: dict[str, Any] = {'n_features': n_features}

    preset_cfg = get_dataset_preset(preset)
    pool_name = str(preset_cfg['pool_preset'])

    pool_cfg = get_pool_preset(pool_name)
    pool = getActivations(**pool_cfg)

    scm_config['activation'] = (
        activation
        if activation is not None
        else pool[int(getRng().integers(0, len(pool)))]
    )
    scm_config['cause_dist'] = (
        cause_dist if cause_dist is not None else preset_cfg['cause_dist']
    )
    scm_config['fixed'] = fixed if fixed is not None else preset_cfg['fixed']
    posthoc_config['p_posthoc'] = (
        p_posthoc if p_posthoc is not None else preset_cfg['p_posthoc']
    )

    scm_config.update(config)
    posthoc_config.update(config)
    posthoc_config['n_features'] = n_features

    scm = SCM(**scm_config)
    x = scm.sample()
    posthoc = Posthoc(**posthoc_config)
    return posthoc(x)
