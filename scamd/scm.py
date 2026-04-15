"""Core structural causal model and post-hoc transformation pipeline."""

from typing import Callable
import numpy as np
import torch
from torch import nn

from scamd.utils import hasConstantColumns, sanityCheck


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


class SCM(nn.Module):
    """Sample synthetic features using an MLP-based structural causal model."""

    def __init__(
        self,
        n_features: int,
        n_causes: int = 10,  # units in initial layer
        n_layers: int = 8,
        n_hidden: int = 32,  # units per layer
        activation: Callable = nn.ReLU,
        sigma_w: float = 1.0,  # for weight initialization
        # feature extraction
        contiguous: bool = False,  # sample adjacent features
        blockwise: bool = True,  # use blockwise dropout
        p_dropout: float = 0.2,  # dropout probability for weights
        # Gaussian noise
        sigma_e: float = 0.01,  # for additive noise
        vary_sigma_e: bool = True,  # allow noise to vary per units
        # misc
        rng: np.random.Generator | None = None,
    ):
        """Initialize SCM sampling modules and random MLP layers."""
        super().__init__()
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.activation = activation
        self.sigma_w = sigma_w
        self.contiguous = contiguous
        self.blockwise = blockwise
        self.p_dropout = p_dropout
        self.sigma_e = sigma_e
        self.vary_sigma_e = vary_sigma_e
        if rng is None:
            rng = np.random.default_rng(0)
        self.rng = rng

        # make sure to have enough hidden units
        self.n_hidden = max(self.n_hidden, 2 * self.n_features)

        # build layers
        layers = [self._buildLayer(n_causes)]
        for _ in range(self.n_layers - 1):
            layers += [self._buildLayer()]
        self.layers = nn.Sequential(*layers)

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

    def _initAllLayers(self):
        """Initialize all linear weight matrices in the network."""
        # init linear weights either with regular droput or blockwise dropout
        for i, block in enumerate(self.layers):
            param = block[0].weight   # type: ignore
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
        param.data *= torch.bernoulli(torch.full_like(param, 1 - p))

    def _initLayerBlockDropout(self, param: torch.Tensor) -> None:
        """Initialize weights in block-diagonal Gaussian submatrices."""
        # blockwise weight dropout for higher dependency between features
        nn.init.zeros_(param)
        max_blocks = np.ceil(np.sqrt(min(param.shape)))
        n_blocks = self.rng.integers(1, max_blocks)
        block_size = [dim // n_blocks for dim in param.shape]
        units_per_block = block_size[0] * block_size[1]
        keep_prob = (n_blocks * units_per_block) / param.numel()
        sigma_w = float(self.sigma_w / (keep_prob**0.5))
        for block in range(n_blocks):
            block_slice = tuple(
                slice(dim * block, dim * (block + 1)) for dim in block_size
            )
            nn.init.normal_(param[block_slice], std=sigma_w)

    def _randomIndices(self, valid: torch.Tensor) -> torch.Tensor:
        valid_idx = np.flatnonzero(valid)
        idx = self.rng.choice(valid_idx, size=self.n_features, replace=False)
        return torch.from_numpy(idx)

    def _contiguousIndices(self, n_units: int, valid: torch.Tensor):
        max_start = n_units - self.n_features + 1
        start_points = self.rng.permutation(max_start)

        # try out starting points
        for start in start_points[: min(max_start, 16)]:
            window = np.arange(start, start + self.n_features)
            if valid[window].all():
                return torch.from_numpy(window)

        # emergency exit
        return self._randomIndices(valid)

    def forward(self, causes: torch.Tensor) -> torch.Tensor | None:
        """Generate one synthetic feature matrix passing sanity checks."""
        self._initAllLayers()

        # pass through each mlp layer
        outputs = [causes]
        for layer in self.layers:
            h = layer(outputs[-1])
            h = torch.where(torch.isfinite(h), h, 0)
            outputs.append(h)
        outputs = outputs[1:]  # remove causes

        # extract features
        outputs = torch.cat(outputs, dim=-1)  # (n, n_units)
        valid = ~hasConstantColumns(outputs)
        if valid.sum() < self.n_features:
            return None

        # choose indices
        n_units = outputs.shape[-1]
        if self.contiguous:
            idx = self._contiguousIndices(n_units, valid)
        else:
            idx = self._randomIndices(valid)
        x = outputs[:, idx]

        # sanity check
        if sanityCheck(x):
            return x
