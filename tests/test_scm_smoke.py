"""Smoke tests covering the SCM pipeline: causes, SCM, posthoc, and the generation API."""

import importlib
import unittest

import numpy as np
import torch
from torch import nn

from scamd import Generator, generateDataset
from scamd.causes import CauseSampler
from scamd.dag import DAGSCM
from scamd.posthoc import Clamp, CensoredFloor, Posthoc
from scamd.scm import SCM, SharedNoiseLayer
from scamd.utils import hasConstantColumns, logUniform, setSeed


class TestSCMSmoke(unittest.TestCase):
    def test_module_imports(self) -> None:
        modules = [
            'scamd.basic',
            'scamd.meta',
            'scamd.causes',
            'scamd.dag',
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
        x = generateDataset(
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
        x = generateDataset(
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
            _ = generateDataset(preset='balanced_realistic')

    def test_generate_dataset_allows_preset_overrides(self) -> None:
        setSeed(24)
        x = generateDataset(
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
        x = generateDataset(
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
        c = gen.sampleCauses(72)
        y = gen(72)
        self.assertEqual(x.shape, (72, 6))
        self.assertEqual(c.shape, (72, 9))
        self.assertEqual(y.shape, (72, 6))
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.isfinite(c).all())
        self.assertTrue(np.isfinite(y).all())

    def test_generator_from_preset(self) -> None:
        setSeed(26)
        gen = Generator.fromPreset(
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

    # ------------------------------------------------------------------
    # New: richer cause distributions
    # ------------------------------------------------------------------

    def test_cause_sampler_new_distributions(self) -> None:
        rng = np.random.default_rng(50)
        for dist_name in (
            '_gamma',
            '_lognormal',
            '_beta',
            '_studentT',
            '_mixtureGaussian',
        ):
            with self.subTest(dist=dist_name):
                sampler = CauseSampler(
                    n_causes=5, dist='normal', fixed_moments=True, rng=rng
                )
                fn = getattr(sampler, dist_name)
                x = fn((120, 5))
                self.assertEqual(x.shape, (120, 5))
                self.assertTrue(torch.isfinite(x).all().item())

    def test_cause_sampler_mixed_uses_new_families(self) -> None:
        # With enough columns the Dirichlet allocation will spread across families
        rng = np.random.default_rng(51)
        sampler = CauseSampler(
            n_causes=20, dist='mixed', fixed_moments=True, rng=rng
        )
        x = sampler.sample(200)
        self.assertEqual(x.shape, (200, 20))
        self.assertTrue(torch.isfinite(x).all().item())

    # ------------------------------------------------------------------
    # New: noise calibration
    # ------------------------------------------------------------------

    def test_noise_calibration_changes_sigma(self) -> None:
        rng = np.random.default_rng(60)
        causes = CauseSampler(
            n_causes=6, dist='normal', fixed_moments=True, rng=rng
        ).sample(300)
        scm = SCM(
            n_features=4,
            n_causes=6,
            n_layers=3,
            n_hidden=16,
            activation=nn.ReLU,
            blockwise=False,
            vary_sigma_e=False,
            calibrate_noise=True,
            calibration_frac=0.1,
            rng=rng,
        )
        x = scm(causes)
        self.assertIsNotNone(x)
        # After calibration, sigma should be a tensor (not the flat float 0.01)
        any_tensor = any(
            isinstance(block[1].sigma, torch.Tensor) for block in scm.layers
        )
        self.assertTrue(
            any_tensor, 'expected calibrated sigma tensors in NoiseLayer'
        )

    def test_noise_calibration_disabled(self) -> None:
        rng = np.random.default_rng(61)
        causes = CauseSampler(
            n_causes=5, dist='normal', fixed_moments=True, rng=rng
        ).sample(100)
        scm = SCM(
            n_features=4,
            n_causes=5,
            n_layers=3,
            n_hidden=12,
            activation=nn.ReLU,
            blockwise=False,
            vary_sigma_e=False,
            calibrate_noise=False,
            rng=rng,
        )
        x = scm(causes)
        self.assertIsNotNone(x)
        self.assertEqual(tuple(x.shape), (100, 4))  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # New: Clamp and CensoredFloor posthoc layers
    # ------------------------------------------------------------------

    def test_clamp_layer_output_shape_and_finite(self) -> None:
        rng = np.random.default_rng(70)
        causes = CauseSampler(
            n_causes=6, dist='normal', fixed_moments=True, rng=rng
        ).sample(100)
        scm = SCM(
            n_features=5,
            n_causes=6,
            n_layers=3,
            n_hidden=12,
            activation=nn.ReLU,
            blockwise=False,
            calibrate_noise=False,
            rng=rng,
        )
        x = scm(causes)
        self.assertIsNotNone(x)
        layer = Clamp(n_in=5, n_out=3, rng=np.random.default_rng(71))
        y = layer(x)
        self.assertEqual(tuple(y.shape), (100, 3))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_censored_floor_layer_output_shape_and_finite(self) -> None:
        rng = np.random.default_rng(72)
        causes = CauseSampler(
            n_causes=6, dist='normal', fixed_moments=True, rng=rng
        ).sample(100)
        scm = SCM(
            n_features=5,
            n_causes=6,
            n_layers=3,
            n_hidden=12,
            activation=nn.ReLU,
            blockwise=False,
            calibrate_noise=False,
            rng=rng,
        )
        x = scm(causes)
        self.assertIsNotNone(x)
        layer = CensoredFloor(n_in=5, n_out=2, rng=np.random.default_rng(73))
        y = layer(x)
        self.assertEqual(tuple(y.shape), (100, 2))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_clamp_values_bounded(self) -> None:
        rng_np = np.random.default_rng(74)
        layer = Clamp(n_in=4, n_out=1, rng=rng_np)
        x = torch.randn(200, 4)
        y = layer(x)
        # values must be non-decreasing range: max - min should be ≤ original range
        self.assertLessEqual(
            (y.max() - y.min()).item(),
            (x.max() - x.min()).item() + 1e-5,
        )

    # ------------------------------------------------------------------
    # New: SharedNoiseLayer
    # ------------------------------------------------------------------

    def test_shared_noise_layer_output_shape(self) -> None:
        idx = torch.tensor([0, 2, 4])
        layer = SharedNoiseLayer(idx, alpha=0.2)
        x = torch.randn(50, 6)
        y = layer(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all().item())
        # unaffected columns should be identical
        self.assertTrue(torch.allclose(y[:, 1], x[:, 1]))
        self.assertTrue(torch.allclose(y[:, 3], x[:, 3]))
        self.assertTrue(torch.allclose(y[:, 5], x[:, 5]))

    # ------------------------------------------------------------------
    # New: DAGSCM
    # ------------------------------------------------------------------

    def test_dagscm_barabasi_albert_shape_and_finite(self) -> None:
        rng = np.random.default_rng(80)
        dag = DAGSCM(
            n_observed=6,
            n_latent=8,
            graph='barabasi_albert',
            m=2,
            activation=nn.ReLU,
            calibrate_noise=True,
            rng=rng,
        )
        x = dag(150)
        self.assertIsNotNone(x)
        self.assertEqual(tuple(x.shape), (150, 6))  # type: ignore[union-attr]
        self.assertTrue(torch.isfinite(x).all().item())  # type: ignore[union-attr]

    def test_dagscm_erdos_renyi_shape_and_finite(self) -> None:
        rng = np.random.default_rng(81)
        dag = DAGSCM(
            n_observed=5,
            n_latent=6,
            graph='erdos_renyi',
            m=2,
            activation=nn.Tanh,
            calibrate_noise=False,
            rng=rng,
        )
        x = dag(100)
        self.assertIsNotNone(x)
        self.assertEqual(tuple(x.shape), (100, 5))  # type: ignore[union-attr]

    def test_generate_dataset_dag_path(self) -> None:
        setSeed(90)
        gen = Generator.fromPreset(
            n_features=7,
            n_causes=5,
            n_layers=3,
            n_hidden=16,
            blockwise=False,
            preset='balanced_realistic',
            use_dag=True,
            rng=np.random.default_rng(90),
        )
        x = gen.sample(120)
        self.assertEqual(x.shape, (120, 7))
        self.assertTrue(np.isfinite(x).all())

    def test_dagscm_invalid_graph_raises(self) -> None:
        with self.assertRaises(ValueError):
            DAGSCM(n_observed=4, n_latent=4, graph='invalid_graph')


if __name__ == '__main__':
    unittest.main()
