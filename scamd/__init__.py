"""Convenience exports for dataset generation and SCM models."""

from .api import generate_dataset
from .plotting import plot_dataset
from .presets import (
    DATASET_PRESETS,
    POOL_PRESETS,
    PRESET_LABELS,
    get_dataset_preset,
    get_pool_preset,
)
from .scm import SCM, Posthoc

__all__ = [
    'SCM',
    'Posthoc',
    'generate_dataset',
    'plot_dataset',
    'POOL_PRESETS',
    'DATASET_PRESETS',
    'PRESET_LABELS',
    'get_pool_preset',
    'get_dataset_preset',
]
