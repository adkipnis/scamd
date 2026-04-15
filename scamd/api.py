"""Public API for synthetic dataset generation."""

from inspect import signature
from typing import Any, Callable

import numpy as np
import torch

from .causes import CauseSampler
from .dag import DAGSCM
from .pool import getActivations
from .presets import getDatasetPreset, getPoolPreset
from .posthoc import Posthoc
from .scm import SCM


class Generator:
    """Bundle CauseSampler, SCM/DAGSCM, and Posthoc into one generation API."""

    def __init__(
        self,
        *,
        causes_config: dict[str, Any],
        scm_config: dict[str, Any],
        posthoc_config: dict[str, Any] | None = None,
        use_dag: bool = False,
        rng: np.random.Generator | None = None,
        max_retries: int = 8,
    ) -> None:
        # max retries
        if max_retries < 0:
            raise ValueError('max_retries must be non-negative')
        self.max_retries = max_retries
        self.use_dag = use_dag

        # configs
        causes_cfg = dict(causes_config)
        scm_cfg = dict(scm_config)
        posthoc_cfg = dict(posthoc_config or {})
        if (
            'n_causes' in causes_cfg
            and 'n_causes' in scm_cfg
            and causes_cfg['n_causes'] != scm_cfg['n_causes']
        ):
            raise ValueError(
                'causes_config.n_causes must match scm_config.n_causes'
            )

        # rng
        if rng is None:
            rng = np.random.default_rng(0)
        self.rng = rng
        causes_cfg['rng'] = self.rng
        scm_cfg['rng'] = self.rng
        posthoc_cfg['rng'] = self.rng

        # main modules
        self.cause_sampler = CauseSampler(**causes_cfg)
        if use_dag:
            dag_cfg = {
                k: v
                for k, v in scm_cfg.items()
                if k
                in (
                    'n_observed',
                    'n_latent',
                    'graph',
                    'm',
                    'activation',
                    'sigma_w',
                    'sigma_e',
                    'calibrate_noise',
                    'calibration_frac',
                    'calibration_n',
                    'rng',
                )
            }
            dag_cfg.setdefault(
                'n_observed',
                scm_cfg.get('n_features', scm_cfg.get('n_observed', 10)),
            )
            self.scm = DAGSCM(**dag_cfg)
        else:
            self.scm = SCM(**scm_cfg)
        self.p_posthoc = float(posthoc_cfg.get('p_posthoc', 0.0))
        self.posthoc = Posthoc(**posthoc_cfg) if self.p_posthoc > 0 else None

    @classmethod
    def fromPreset(
        cls,
        *,
        n_features: int,
        n_causes: int,
        n_layers: int,
        n_hidden: int,
        blockwise: bool,
        contiguous: bool = False,
        preset: str = 'balanced_realistic',
        activation: Callable | None = None,
        p_posthoc: float | None = None,
        cause_dist: str | None = None,
        fixed: bool | None = None,
        use_dag: bool = False,
        dag_m: int | None = None,
        dag_graph: str | None = None,
        rng: np.random.Generator | None = None,
        max_retries: int = 8,
        **config: Any,
    ) -> 'Generator':
        """Build cause/SCM/posthoc configs from a named preset."""
        shared_rng = np.random.default_rng(0) if rng is None else rng
        preset_cfg = getDatasetPreset(preset)
        pool_name = str(preset_cfg['pool_preset'])
        pool_cfg = getPoolPreset(pool_name)
        pool = getActivations(**pool_cfg, rng=shared_rng)

        causes_config: dict[str, Any] = {
            'n_causes': n_causes,
            'dist': cause_dist
            if cause_dist is not None
            else preset_cfg['cause_dist'],
            'fixed_moments': fixed
            if fixed is not None
            else preset_cfg['fixed'],
        }
        scm_config: dict[str, Any] = {
            'n_features': n_features,
            'n_causes': n_causes,
            'n_layers': n_layers,
            'n_hidden': n_hidden,
            'blockwise': blockwise,
            'contiguous': contiguous,
            'activation': (
                activation
                if activation is not None
                else pool[int(shared_rng.integers(0, len(pool)))]
            ),
        }
        posthoc_config: dict[str, Any] = {
            'n_features': n_features,
            'p_posthoc': p_posthoc
            if p_posthoc is not None
            else preset_cfg['p_posthoc'],
        }

        # when using a DAG-SCM, sample graph topology parameters unless overridden
        if use_dag:
            scm_config['m'] = dag_m if dag_m is not None else int(
                shared_rng.integers(1, 6)
            )
            scm_config['graph'] = dag_graph if dag_graph is not None else str(
                shared_rng.choice(['barabasi_albert', 'erdos_renyi'])
            )

        causes_keys = set(signature(CauseSampler).parameters)
        scm_keys = set(signature(SCM).parameters)
        posthoc_keys = set(signature(Posthoc).parameters)
        for key, value in config.items():
            if key in causes_keys:
                causes_config[key] = value
            if key in scm_keys:
                scm_config[key] = value
            if key in posthoc_keys:
                posthoc_config[key] = value

        return cls(
            causes_config=causes_config,
            scm_config=scm_config,
            posthoc_config=posthoc_config,
            use_dag=use_dag,
            rng=shared_rng,
            max_retries=max_retries,
        )

    @torch.inference_mode()
    def sample(
        self, n_samples: int, return_numpy: bool = True
    ) -> np.ndarray | torch.Tensor:
        """Sample causes, transform with SCM/DAGSCM, then optional Posthoc."""
        for _ in range(self.max_retries):
            if self.use_dag:
                x = self.scm(n_samples)
            else:
                causes = self.cause_sampler.sample(n_samples)
                x = self.scm(causes)
            if x is None:
                continue
            if self.posthoc is not None and self.p_posthoc > 0:
                x = self.posthoc(x)
            if return_numpy:
                return x.detach().cpu().numpy()
            return x

        raise RuntimeError(
            f'Generator.sample failed to produce a valid sample in {self.max_retries} attempts'
        )

    def __call__(
        self, n_samples: int, return_numpy: bool = True
    ) -> np.ndarray | torch.Tensor:
        return self.sample(n_samples=n_samples, return_numpy=return_numpy)

    @torch.inference_mode()
    def sampleCauses(
        self, n_samples: int, return_numpy: bool = True
    ) -> np.ndarray | torch.Tensor:
        """Sample only root causes from the bundled cause sampler."""
        x = self.cause_sampler.sample(n_samples)
        if return_numpy:
            return x.detach().cpu().numpy()
        return x


def generateDataset(
    *,
    n_samples: int,
    n_features: int,
    n_causes: int,
    n_layers: int,
    n_hidden: int,
    blockwise: bool,
    contiguous: bool = False,
    preset: str = 'balanced_realistic',
    activation: Callable | None = None,
    p_posthoc: float | None = None,
    cause_dist: str | None = None,
    fixed: bool | None = None,
    use_dag: bool = False,
    dag_m: int | None = None,
    dag_graph: str | None = None,
    rng: np.random.Generator | None = None,
    **config: Any,
) -> np.ndarray:
    """Generate one X-only synthetic dataset.

    A convenience wrapper around ``Generator.fromPreset`` that returns a
    plain NumPy array with shape ``(n_samples, n_features)``.
    """
    generator = Generator.fromPreset(
        n_features=n_features,
        n_causes=n_causes,
        n_layers=n_layers,
        n_hidden=n_hidden,
        blockwise=blockwise,
        contiguous=contiguous,
        preset=preset,
        activation=activation,
        p_posthoc=p_posthoc,
        cause_dist=cause_dist,
        fixed=fixed,
        use_dag=use_dag,
        dag_m=dag_m,
        dag_graph=dag_graph,
        rng=rng,
        **config,
    )
    return generator.sample(n_samples=n_samples, return_numpy=True)
