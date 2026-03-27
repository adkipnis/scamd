"""Convenience exports for dataset generation and SCM models."""

from .api import generate_dataset
from .scm import SCM, Posthoc

__all__ = ['SCM', 'Posthoc', 'generate_dataset']
