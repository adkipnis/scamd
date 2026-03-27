import importlib
import unittest

import numpy as np
import torch
from torch import nn

from scamd import Generator, generate_dataset
from scamd.causes import CauseSampler
from scamd.posthoc import Posthoc
from scamd.scm import SCM
from scamd.utils import hasConstantColumns, logUniform, setSeed


class TestSCMSmoke(unittest.TestCase):
    def test_module_imports(self) -> None:
        modules = [
            'scamd.basic',
            'scamd.meta',
            'scamd.causes',
            'scamd.posthoc',
            'scamd.gp',
            'scamd.pool',
            'scamd.scm',
            'scamd.api',
        ]
        for module_name in modules:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module)

    def test_scm_sample_shape_and_finite(self) -> None:
        setSeed(11)
        causes = CauseSampler(
            n_causes=8,
            dist='normal',
            fixed_moments=True,
        ).sample(128)
        scm = SCM(
            n_features=6,
            n_causes=8,
            n_layers=4,
            n_hidden=16,
            activation=nn.Tanh,
            blockwise=False,
            vary_sigma_e=False,
        )
        x = scm(causes)
        self.assertIsNotNone(x)
        x = x  # type: ignore[assignment]
        self.assertEqual(tuple(x.shape), (128, 6))
        self.assertTrue(torch.isfinite(x).all().item())
        self.assertFalse(hasConstantColumns(x).any().item())

    def test_posthoc_output_shape_finite(self) -> None:
        setSeed(12)
        causes = CauseSampler(
            n_causes=7,
            dist='uniform',
            fixed_moments=True,
        ).sample(96)
        scm = SCM(
            n_features=5,
            n_causes=7,
            n_layers=3,
            n_hidden=12,
            activation=nn.ReLU,
            blockwise=False,
            vary_sigma_e=False,
        )
        x = scm(causes)
        self.assertIsNotNone(x)
        x = x  # type: ignore[assignment]
        ph = Posthoc(n_features=5, p_posthoc=0.4)
        y = ph(x)
        self.assertEqual(y.shape, (96, 5))
        self.assertTrue(torch.isfinite(y).all().item())
        self.assertFalse(hasConstantColumns(y).any().item())

    def test_reproducible_with_seed(self) -> None:
        cfg = {
            'n_features': 4,
            'n_causes': 6,
            'n_layers': 3,
            'n_hidden': 10,
            'activation': nn.Tanh,
            'blockwise': False,
            'vary_sigma_e': False,
        }
        setSeed(123)
        c1 = CauseSampler(
            n_causes=6, dist='normal', fixed_moments=True
        ).sample(64)
        x1 = SCM(**cfg)(c1)
        setSeed(123)
        c2 = CauseSampler(
            n_causes=6, dist='normal', fixed_moments=True
        ).sample(64)
        x2 = SCM(**cfg)(c2)
        self.assertIsNotNone(x1)
        self.assertIsNotNone(x2)
        x1 = x1  # type: ignore[assignment]
        x2 = x2  # type: ignore[assignment]
        self.assertTrue(torch.allclose(x1, x2))

    def test_log_uniform_rng_first_scalar_and_vector(self) -> None:
        setSeed(7)
        rng = np.random.default_rng(7)
        scalar = logUniform(rng, 0.1, 1.0)
        vec = logUniform(rng, 2.0, 20.0, size=(5,), round=True)
        self.assertTrue(np.isscalar(scalar))
        self.assertEqual(vec.shape, (5,))
        self.assertTrue(np.issubdtype(vec.dtype, np.integer))

    def test_generate_dataset_api_shape_and_finite(self) -> None:
        setSeed(21)
        x = generate_dataset(
            n_samples=80,
            n_features=7,
            n_causes=10,
            n_layers=5,
            n_hidden=24,
            blockwise=True,
            cause_dist='mixed',
            activation=nn.SiLU,
        )
        self.assertEqual(x.shape, (80, 7))
        self.assertTrue(np.isfinite(x).all())

    def test_generate_dataset_with_preset(self) -> None:
        setSeed(22)
        x = generate_dataset(
            n_samples=90,
            n_features=9,
            n_causes=12,
            n_layers=6,
            n_hidden=36,
            blockwise=True,
            preset='balanced_realistic',
        )
        self.assertEqual(x.shape, (90, 9))
        self.assertTrue(np.isfinite(x).all())

    def test_generate_dataset_requires_explicit_scm_size(self) -> None:
        setSeed(23)
        with self.assertRaises(TypeError):
            _ = generate_dataset(preset='balanced_realistic')

    def test_generate_dataset_allows_preset_overrides(self) -> None:
        setSeed(24)
        x = generate_dataset(
            n_samples=72,
            n_features=6,
            n_causes=9,
            n_layers=4,
            n_hidden=20,
            blockwise=False,
            preset='smooth_stable',
            fixed=False,
            p_posthoc=0.5,
        )
        self.assertEqual(x.shape, (72, 6))
        self.assertTrue(np.isfinite(x).all())

    def test_generate_dataset_contiguous_passthrough(self) -> None:
        setSeed(27)
        x = generate_dataset(
            n_samples=64,
            n_features=6,
            n_causes=9,
            n_layers=4,
            n_hidden=20,
            blockwise=False,
            contiguous=True,
            preset='smooth_stable',
        )
        self.assertEqual(x.shape, (64, 6))
        self.assertTrue(np.isfinite(x).all())

    def test_generator_api_samples_data_and_causes(self) -> None:
        setSeed(25)
        gen = Generator(
            causes_config={
                'n_causes': 9,
                'dist': 'mixed',
                'fixed_moments': False,
            },
            scm_config={
                'n_features': 6,
                'n_causes': 9,
                'n_layers': 4,
                'n_hidden': 20,
                'blockwise': False,
            },
            posthoc_config={
                'n_features': 6,
                'p_posthoc': 0.5,
            },
        )
        x = gen.sample(72)
        c = gen.sample_causes(72)
        y = gen(72)
        self.assertEqual(x.shape, (72, 6))
        self.assertEqual(c.shape, (72, 9))
        self.assertEqual(y.shape, (72, 6))
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.isfinite(c).all())
        self.assertTrue(np.isfinite(y).all())

    def test_generator_from_preset(self) -> None:
        setSeed(26)
        gen = Generator.from_preset(
            n_features=6,
            n_causes=9,
            n_layers=4,
            n_hidden=20,
            blockwise=False,
            contiguous=True,
            preset='smooth_stable',
            fixed=False,
            p_posthoc=0.5,
        )
        x = gen.sample(72)
        self.assertEqual(x.shape, (72, 6))
        self.assertTrue(np.isfinite(x).all())

    def test_max_retries_raises(self) -> None:
        setSeed(33)
        gen = Generator(
            causes_config={
                'n_causes': 6,
            },
            scm_config={
                'n_features': 4,
                'n_causes': 6,
            },
            posthoc_config={
                'n_features': 4,
                'p_posthoc': 0.0,
            },
            max_retries=0,
        )
        with self.assertRaises(RuntimeError):
            _ = gen.sample(64)


if __name__ == '__main__':
    unittest.main()
