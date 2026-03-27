"""Core structural causal model and post-hoc transformation pipeline."""

from typing import Callable
import numpy as np
import torch
from torch import nn

from .causes import CauseSampler
from .posthoc import getPosthocLayers
from .utils import standardize, checkConstant, getRng

phl = getPosthocLayers()


class NoiseLayer(nn.Module):
    """Add elementwise Gaussian noise with a configurable scale."""

    def __init__(self, sigma: float | torch.Tensor):
        """Store the noise scale used during forward passes."""
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input perturbed by i.i.d. Gaussian noise."""
        noise = torch.randn_like(x) * self.sigma
        return x + noise


def sanityCheck(x: torch.Tensor) -> bool:
    """Check that no feature column is constant across samples."""
    x_np = x.detach().cpu().numpy()
    okay = not checkConstant(x_np).any()
    return okay


class SCM(nn.Module):
    """Sample synthetic features using an MLP-based structural causal model."""

    def __init__(
        self,
        # data dims
        n_samples: int,
        n_features: int,
        # causes
        n_causes: int = 10,  # number of units in initial layer
        cause_dist: str = 'uniform',  # [mixed, normal, uniform]
        fixed: bool = False,  # fixed moments of causes
        # MLP architecture
        n_layers: int = 8,
        n_hidden: int = 32,
        activation: Callable = nn.Tanh,
        # weight initialization and feature extraction
        sigma_w: float = 1.0,  # for weight initialization
        contiguous: bool = False,  # sample adjacent features
        blockwise: bool = True,  # use blockwise dropout
        p_dropout: float = 0.1,  # dropout probability for weights
        # Gaussian noise
        sigma_e: float = 0.01,  # for additive noise
        vary_sigma_e: bool = True,  # allow noise to vary per units
        **kwargs,
    ):
        """Initialize SCM sampling modules and random MLP layers."""
        super().__init__()
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_causes = n_causes
        self.cause_dist = cause_dist
        self.fixed = fixed
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.activation = activation
        self.sigma_w = sigma_w
        self.contiguous = contiguous
        self.blockwise = blockwise
        self.p_dropout = p_dropout
        self.sigma_e = sigma_e
        self.vary_sigma_e = vary_sigma_e
        self.max_retries = int(kwargs.get('max_retries', 64))

        # make sure to have enough hidden units
        self.n_hidden = max(self.n_hidden, 2 * self.n_features)

        # init sampler for root nodes
        self.cs = CauseSampler(
            self.n_samples,
            self.n_causes,
            dist=self.cause_dist,
            fixed=self.fixed,
        )

        # build layers
        layers = [self._buildLayer(self.n_causes)]
        for _ in range(self.n_layers - 1):
            layers += [self._buildLayer()]
        self.layers = nn.Sequential(*layers)

        # initialize weights
        with torch.no_grad():
            self._initLayers()

    def _buildLayer(self, input_dim: int = 0) -> nn.Module:
        """Create one affine-noise-activation block."""
        # Affine() ->  AdditiveNoise() -> Activation()
        if input_dim == 0:
            input_dim = self.n_hidden
        affine_layer = nn.Linear(input_dim, self.n_hidden)
        sigma_e = self.sigma_e
        if self.vary_sigma_e:
            sigma_e = (torch.randn((self.n_hidden,)) * self.sigma_e).abs()
        noise_layer = NoiseLayer(sigma_e)
        return nn.Sequential(affine_layer, noise_layer, self.activation())

    def _initLayers(self):
        """Initialize all linear weight matrices in the network."""
        # init linear weights either with regular droput or blockwise dropout
        for i, block in enumerate(self.layers):
            param = block[0].weight
            if self.blockwise:
                self._initLayerBlockDropout(param)
            else:
                self._initLayer(param, i > 0)

    def _initLayer(
        self, param: torch.Tensor, use_dropout: bool = True
    ) -> None:
        """Sample dense Gaussian weights with optional Bernoulli dropout."""
        p = self.p_dropout if use_dropout else 0.0
        p = min(p, 0.99)
        sigma_w = self.sigma_w / ((1 - p) ** 0.5)
        nn.init.normal_(param, std=sigma_w)
        param *= torch.bernoulli(torch.full_like(param, 1 - p))

    def _initLayerBlockDropout(self, param: torch.Tensor) -> None:
        """Initialize weights in block-diagonal Gaussian submatrices."""
        # blockwise weight dropout for higher dependency between features
        nn.init.zeros_(param)
        max_blocks = np.ceil(np.sqrt(min(param.shape)))
        n_blocks = getRng().integers(1, max_blocks)
        block_size = [dim // n_blocks for dim in param.shape]
        units_per_block = block_size[0] * block_size[1]
        keep_prob = (n_blocks * units_per_block) / param.numel()
        sigma_w = self.sigma_w / (keep_prob**0.5)
        for block in range(n_blocks):
            block_slice = tuple(
                slice(dim * block, dim * (block + 1)) for dim in block_size
            )
            nn.init.normal_(param[block_slice], std=sigma_w)

    def sample(self) -> torch.Tensor:
        """Generate one synthetic feature matrix passing sanity checks."""
        for _ in range(self.max_retries):
            causes = self.cs.sample()  # (seq_len, num_causes)

            # pass through each mlp layer
            outputs = [causes]
            for layer in self.layers:
                h = layer(outputs[-1])
                h = torch.where(h.isnan() | h.abs().isinf(), 0, h)
                outputs.append(h)
            outputs = outputs[1:]  # remove causes

            # extract features
            outputs = torch.cat(outputs, dim=-1)  # (n, units)
            n_units = outputs.shape[-1]
            if self.contiguous:
                start = getRng().integers(0, n_units - self.n_features + 1)
                perm = start + torch.randperm(self.n_features)
            else:
                perm = torch.randperm(n_units)
            indices = perm[: self.n_features]
            x = outputs[:, indices]
            if sanityCheck(x):
                return x
        raise RuntimeError(
            f'SCM.sample failed to produce a valid sample in {self.max_retries} attempts'
        )


class Posthoc(nn.Module):
    """Apply optional post-hoc feature transformations to SCM outputs."""

    def __init__(
        self,
        n_features: int,
        p_posthoc: float = 0.2,  # probability of posthoc transformation
        standardize: bool = True,
        **kwargs,
    ):
        """Initialize a random set of post-hoc transformation layers."""
        super().__init__()
        self.standardize = standardize

        # posthoc transformations
        self.n_features = n_features
        self.n_posthoc = getRng().binomial(n_features, p_posthoc)
        layers = []
        for _ in range(self.n_posthoc):
            cfg = {
                'n_in': n_features,
                'n_out': getRng().integers(1, 3),
                'standardize': True,
                # TODO levels, sigma
            }
            layer = getRng().choice(phl)

            layers.append(layer(**cfg))
        self.transformations = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> np.ndarray:
        """Transform, subsample, and optionally standardize feature columns."""
        if self.n_posthoc > 0:
            out = []
            for t in self.transformations:
                h = t(x)
                if sanityCheck(h):
                    out.append(h)
            if out:
                z = torch.cat(out, dim=-1)
                x = torch.cat([x, z], dim=-1)
                idx = torch.randperm(x.shape[-1])[: self.n_features]
                x = x[..., idx]
        x = x.detach().cpu().numpy()
        if self.standardize:
            x = standardize(x, axis=0)
        return x
