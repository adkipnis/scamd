"""Named presets shared by demos and dataset generation API."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


POOL_PRESETS: dict[str, dict[str, Any]] = {
    'smooth_stable': {
        'n_gp': 8,
        'n_random_choice': 0,
        'random_scale': False,
        'gp_types': ('se', 'matern'),
        'gp_type_probs': (0.7, 0.3),
        'allow_nested_random_choice': False,
    },
    'balanced_realistic': {
        'n_gp': 16,
        'n_random_choice': 3,
        'random_scale': True,
        'gp_type_probs': (0.35, 0.25, 0.40),
        'allow_nested_random_choice': False,
    },
    'high_variability': {
        'n_gp': 20,
        'n_random_choice': 5,
        'random_scale': True,
        'gp_type_probs': (0.2, 0.25, 0.55),
        'allow_nested_random_choice': False,
    },
}


DATASET_PRESETS: dict[str, dict[str, Any]] = {
    'smooth_stable': {
        'p_posthoc': 0.10,
        'cause_dist': 'uniform',
        'fixed': True,
        'pool_preset': 'smooth_stable',
    },
    'balanced_realistic': {
        'p_posthoc': 0.35,
        'cause_dist': 'mixed',
        'fixed': False,
        'pool_preset': 'balanced_realistic',
    },
    'high_variability': {
        'p_posthoc': 0.20,
        'cause_dist': 'mixed',
        'fixed': False,
        'pool_preset': 'high_variability',
    },
}


PRESET_LABELS: dict[str, str] = {
    'balanced_realistic': 'Balanced Realistic',
    'smooth_stable': 'Smooth + Stable',
    'high_variability': 'High Variability',
}


def get_pool_preset(name: str) -> dict[str, Any]:
    """Return a deep copy of one activation-pool preset."""
    if name not in POOL_PRESETS:
        options = ', '.join(sorted(POOL_PRESETS))
        raise ValueError(
            f'unknown pool preset {name!r}; choose one of: {options}'
        )
    return deepcopy(POOL_PRESETS[name])


def get_dataset_preset(name: str) -> dict[str, Any]:
    """Return a deep copy of one dataset-generation preset."""
    if name not in DATASET_PRESETS:
        options = ', '.join(sorted(DATASET_PRESETS))
        raise ValueError(
            f'unknown dataset preset {name!r}; choose one of: {options}'
        )
    return deepcopy(DATASET_PRESETS[name])
