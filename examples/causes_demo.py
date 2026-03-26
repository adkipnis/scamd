"""Exercise the cause sampler across supported distributions.

This script runs several CauseSampler configurations as a quick sanity check.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.causes import CauseSampler


if __name__ == '__main__':
    config = {
        'n_samples': 100,
        'n_causes': 8,
        'dist': 'normal',
        'fixed': True,
    }
    x = CauseSampler(**config).sample()

    config.update({'fixed': False})
    x = CauseSampler(**config).sample()

    config.update({'dist': 'uniform'})
    x = CauseSampler(**config).sample()

    config.update({'dist': 'mixed', 'fixed': True})
    x = CauseSampler(**config).sample()

    config.update({'dist': 'mixed', 'fixed': False})
    x = CauseSampler(**config).sample()
