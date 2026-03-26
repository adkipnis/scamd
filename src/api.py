"""Public API for synthetic dataset generation."""

from typing import Any

import numpy as np

from .scm import Posthoc, SCM


def generate_dataset(**config: Any) -> np.ndarray:
    """Generate one standardized X-only synthetic dataset."""
    scm = SCM(**config)
    x = scm.sample()
    posthoc = Posthoc(**config)
    return posthoc(x)
