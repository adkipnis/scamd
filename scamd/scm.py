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


class SharedNoiseLayer(nn.Module):
    """Add a shared latent noise component to a random subset of features.

    One noise vector z ~ N(0, 1) is drawn per sample and mixed into
    ``feature_indices`` with weight ``alpha``, injecting residual correlation
    that is not explained by the causal MLP structure.
    """

    def __init__(self, feature_indices: torch.Tensor, alpha: float):
        super().__init__()
        self.register_buffer('feature_indices', feature_indices)
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        z = torch.randn(n, device=x.device)          # (n,)
        noise = z.unsqueeze(-1) * self.alpha          # (n, 1) broadcast
        x = x.clone()
        x[:, self.feature_indices] = x[:, self.feature_indices] + noise
        return x


class MarginalTransformLayer(nn.Module):
    """Apply per-feature monotone transforms to diversify marginal shapes.

    Each feature independently receives one of: identity, signed power
    ``sign(x)|x|^p``, or log-compression ``sign(x)log(1+|x|)``.  Mixing
    transform types across features that share latent structure preserves
    rank correlation (Spearman) while reducing Pearson, matching the
    Spearman > Pearson pattern common in real datasets.
    """

    _TYPES = ('identity', 'signed_power', 'log1p')

    def __init__(self, n_features: int, rng: np.random.Generator):
        super().__init__()
        types = rng.choice(self._TYPES, size=n_features)
        powers = rng.uniform(0.3, 2.5, size=n_features).tolist()
        self.types: list[str] = types.tolist()
        self.powers: list[float] = powers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cols = []
        for j, (t, p) in enumerate(zip(self.types, self.powers)):
            col = x[:, j]
            if t == 'signed_power':
                col = col.sign() * col.abs().clamp(max=1e3).pow(p)
            elif t == 'log1p':
                col = col.sign() * (col.abs() + 1.0).log()
            cols.append(col)
        return torch.stack(cols, dim=1)


class FactorLayer(nn.Module):
    """Inject a low-rank factor structure to increase inter-feature correlation.

    Samples ``n_factors`` shared latent factors z ~ N(0, I) per forward pass
    and mixes them into the feature matrix via fixed random loadings W,
    inducing a rank-``n_factors`` correlation block on top of the SCM output.
    """

    def __init__(self, n_features: int, n_factors: int, alpha: float):
        super().__init__()
        W = torch.randn(n_features, n_factors) / (n_factors ** 0.5)
        self.register_buffer('W', W)
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.randn(x.shape[0], self.W.shape[1], device=x.device)  # (n, k)
        return x + self.alpha * (z @ self.W.T)                          # (n, d)


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
        # noise calibration
        calibrate_noise: bool = True,  # scale sigma_e to signal IQR
        calibration_frac: float = 0.1,  # noise = calibration_frac * IQR
        calibration_n: int = 256,  # pilot batch size for calibration
        # shared noise groups
        p_shared_noise: float = 0.5,  # probability of adding any shared noise
        # per-feature monotone marginal transforms
        p_marginal_transform: float = 0.5,  # probability of adding MarginalTransformLayer
        # low-rank factor injection
        p_factor: float = 0.5,  # probability of adding a FactorLayer
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
        self.calibrate_noise = calibrate_noise
        self.calibration_frac = calibration_frac
        self.calibration_n = calibration_n
        self.p_shared_noise = p_shared_noise
        self.p_marginal_transform = p_marginal_transform
        self.p_factor = p_factor
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

        # build shared noise layers (applied after feature extraction)
        self.shared_noise_layers = self._buildSharedNoiseLayers()

        # optional per-feature monotone marginal transforms (applied before shared noise)
        self.marginal_transform = self._buildMarginalTransform()

        # optional low-rank factor injection (applied before shared noise)
        self.factor_layer = self._buildFactorLayer()

    def _buildSharedNoiseLayers(self) -> nn.ModuleList:
        """Create 0–3 shared noise components over random feature subsets."""
        layers = []
        if self.n_features < 2:
            return nn.ModuleList(layers)
        if self.p_shared_noise > 0 and self.rng.random() < self.p_shared_noise:
            n_groups = int(self.rng.integers(1, 4))
            for _ in range(n_groups):
                group_size = int(self.rng.integers(2, self.n_features + 1))
                indices = torch.from_numpy(
                    self.rng.choice(
                        self.n_features, size=group_size, replace=False
                    )
                )
                alpha = float(self.rng.uniform(0.05, 0.4))
                layers.append(SharedNoiseLayer(indices, alpha))
        return nn.ModuleList(layers)

    def _buildMarginalTransform(self) -> MarginalTransformLayer | None:
        """Create an optional per-feature monotone transform layer."""
        if self.p_marginal_transform <= 0 or self.rng.random() >= self.p_marginal_transform:
            return None
        return MarginalTransformLayer(self.n_features, self.rng)

    def _buildFactorLayer(self) -> FactorLayer | None:
        """Create an optional low-rank factor injection layer."""
        if self.n_features < 2 or self.p_factor <= 0:
            return None
        if self.rng.random() >= self.p_factor:
            return None
        n_factors = int(self.rng.integers(1, max(2, self.n_features // 2 + 1)))
        alpha = float(self.rng.uniform(0.1, 0.8))
        return FactorLayer(self.n_features, n_factors, alpha)

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

    def _calibrateNoise(self, causes: torch.Tensor) -> None:
        """Scale each NoiseLayer's sigma so noise ≈ calibration_frac * signal IQR."""
        # temporarily zero out noise
        noise_layers = [block[1] for block in self.layers]
        for nl in noise_layers:
            nl.sigma = (
                torch.zeros_like(nl.sigma)
                if isinstance(nl.sigma, torch.Tensor)
                else 0.0
            )

        with torch.no_grad():
            h = causes
            for i, layer in enumerate(self.layers):
                h = layer(h)
                h = torch.where(torch.isfinite(h), h, torch.zeros_like(h))
                q75 = torch.quantile(h, 0.75, dim=0)
                q25 = torch.quantile(h, 0.25, dim=0)
                iqr = (q75 - q25).clamp(min=1e-6)
                noise_layers[i].sigma = (self.calibration_frac * iqr).detach()

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

        # calibrate noise scales on first forward pass
        if self.calibrate_noise and not getattr(
            self, '_noise_calibrated', False
        ):
            pilot = (
                causes[: self.calibration_n]
                if causes.shape[0] >= self.calibration_n
                else causes
            )
            self._calibrateNoise(pilot)
            self._noise_calibrated = True

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

        # apply low-rank factor injection
        if self.factor_layer is not None:
            x = self.factor_layer(x)

        # apply shared noise groups
        for snl in self.shared_noise_layers:
            x = snl(x)

        # apply per-feature monotone marginal transforms
        if self.marginal_transform is not None:
            x = self.marginal_transform(x)

        # sanity check
        if sanityCheck(x):
            return x
