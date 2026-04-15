"""Stress test: generate_dataset with 10k random hyperparameter draws (opt-in via env var)."""

import os
import unittest

import numpy as np
from torch import nn

from scamd import generateDataset
from scamd.utils import setSeed


def _sample_generate_dataset_kwargs(rng: np.random.Generator) -> dict:
    n_features = int(rng.integers(2, 33))
    n_causes_low = max(3, n_features)
    n_causes_high = max(n_causes_low + 1, (n_features * 4) + 1)
    n_causes = int(rng.integers(n_causes_low, n_causes_high))
    n_layers = int(rng.integers(1, 9))
    n_hidden_min = max(2 * n_features, 8)
    n_hidden = int(rng.integers(n_hidden_min, (n_hidden_min * 3) + 1))

    activations = (nn.Tanh, nn.ReLU, nn.SiLU, nn.ELU, nn.GELU)
    presets = ('smooth_stable', 'balanced_realistic', 'high_variability')
    cause_dists = ('normal', 'uniform', 'mixed')

    return {
        'n_samples': int(rng.integers(16, 257)),
        'n_features': n_features,
        'n_causes': n_causes,
        'n_layers': n_layers,
        'n_hidden': n_hidden,
        'blockwise': bool(rng.integers(0, 2)),
        'preset': str(rng.choice(presets)),
        'activation': rng.choice(activations),
        'p_posthoc': float(rng.uniform(0.0, 0.5)),
        'cause_dist': str(rng.choice(cause_dists)),
        'fixed': bool(rng.integers(0, 2)),
    }


class TestGenerateDatasetStress(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv('SCAMD_RUN_STRESS_TEST', '0') == '1',
        'set SCAMD_RUN_STRESS_TEST=1 to run stress tests',
    )
    def test_generate_dataset_random_hyperparams_10k(self) -> None:
        iterations = int(os.getenv('SCAMD_STRESS_ITERS', '10000'))
        rng = np.random.default_rng(20260327)

        n_errors = 0
        error_examples: list[str] = []

        for i in range(iterations):
            kwargs = _sample_generate_dataset_kwargs(rng)
            setSeed(int(rng.integers(0, 2**31 - 1)))
            try:
                x = generateDataset(**kwargs)
                if x.shape != (kwargs['n_samples'], kwargs['n_features']):
                    raise AssertionError(
                        f'unexpected shape: {x.shape}, expected '
                        f"({kwargs['n_samples']}, {kwargs['n_features']})"
                    )
                if not np.isfinite(x).all():
                    raise AssertionError(
                        'non-finite values in generated dataset'
                    )
            except Exception as exc:
                n_errors += 1
                if len(error_examples) < 8:
                    error_examples.append(
                        f'iter={i}, error={type(exc).__name__}: {exc}, kwargs={kwargs}'
                    )

        self.assertEqual(
            n_errors,
            0,
            msg=(
                f'{n_errors} / {iterations} iterations failed. '
                'First failures:\n' + '\n'.join(error_examples)
            ),
        )


if __name__ == '__main__':
    unittest.main()
