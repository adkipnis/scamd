import importlib
import unittest

import numpy as np
import torch
from torch import nn

from src.scm import Posthoc, SCM
from src.utils import checkConstant, getRng, logUniform, setSeed


class TestSCMSmoke(unittest.TestCase):
    def test_module_imports(self) -> None:
        modules = [
            "src.basic",
            "src.meta",
            "src.causes",
            "src.posthoc",
            "src.gp",
            "src.activations",
            "src.scm",
        ]
        for module_name in modules:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module)

    def test_scm_sample_shape_and_finite(self) -> None:
        setSeed(11)
        scm = SCM(
            n_samples=128,
            n_features=6,
            n_causes=8,
            cause_dist="normal",
            fixed=True,
            n_layers=4,
            n_hidden=16,
            activation=nn.Tanh,
            blockwise=False,
            vary_sigma_e=False,
        )
        x = scm.sample()
        self.assertEqual(tuple(x.shape), (128, 6))
        self.assertTrue(torch.isfinite(x).all().item())
        self.assertFalse(checkConstant(x.detach().numpy()).any())

    def test_posthoc_output_shape_finite(self) -> None:
        setSeed(12)
        scm = SCM(
            n_samples=96,
            n_features=5,
            n_causes=7,
            cause_dist="uniform",
            fixed=True,
            n_layers=3,
            n_hidden=12,
            activation=nn.ReLU,
            blockwise=False,
            vary_sigma_e=False,
        )
        x = scm.sample()
        ph = Posthoc(n_features=5, p_posthoc=0.4)
        y = ph(x)
        self.assertEqual(y.shape, (96, 5))
        self.assertTrue(np.isfinite(y).all())
        self.assertFalse(checkConstant(y).any())

    def test_reproducible_with_seed(self) -> None:
        cfg = {
            "n_samples": 64,
            "n_features": 4,
            "n_causes": 6,
            "cause_dist": "normal",
            "fixed": True,
            "n_layers": 3,
            "n_hidden": 10,
            "activation": nn.Tanh,
            "blockwise": False,
            "vary_sigma_e": False,
        }
        setSeed(123)
        x1 = SCM(**cfg).sample()
        setSeed(123)
        x2 = SCM(**cfg).sample()
        self.assertTrue(torch.allclose(x1, x2))

    def test_log_uniform_rng_first_scalar_and_vector(self) -> None:
        setSeed(7)
        rng = getRng()
        scalar = logUniform(rng, 0.1, 1.0)
        vec = logUniform(rng, 2.0, 20.0, size=(5,), round=True)
        self.assertTrue(np.isscalar(scalar))
        self.assertEqual(vec.shape, (5,))
        self.assertTrue(np.issubdtype(vec.dtype, np.integer))


if __name__ == "__main__":
    unittest.main()
