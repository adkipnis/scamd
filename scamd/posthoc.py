"""Deterministic and stochastic post-hoc feature transformations."""

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D
import numpy as np
from .meta import Standardizer
from .utils import sanityCheck


# --- deterministic post-hoc layers


class Base(nn.Module):
    """Base layer that mixes input features into post-hoc outputs."""

    def __init__(self, n_in: int, n_out: int, standardize: bool = False):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.standardize = standardize
        if standardize:
            self.standardizer = Standardizer()
        alpha = torch.ones(n_in)
        self.w = (
            D.Dirichlet(alpha).sample((n_out, self.nParam)).permute(2, 0, 1)
        )

    @property
    def nParam(self) -> int:
        """Number of parameters per output channel."""
        return 1

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Optionally standardize and apply learned random mixing weights."""
        if self.standardize:
            x = self.standardizer(x)
        x = torch.einsum('...nd,dap->...nap', x, self.w)
        return x


class Threshold(Base):
    """Binarize each mixed feature at zero."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)[..., 0]
        x = (x > 0).float()
        return x


class MultiThreshold(Base):
    """Map each mixed feature to an ordinal level via multiple thresholds."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        standardize: bool = False,
        levels: int = 3,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(n_in, n_out, standardize)
        self.levels = levels
        if rng is None:
            rng = np.random.default_rng(0)
        self.tau = np.sort(rng.normal(size=levels - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Count how many sampled thresholds each value exceeds."""
        x = self.preprocess(x)[..., 0]
        y = torch.zeros_like(x)
        for t in self.tau:
            y = y + (x > t)
        return y


class QuantileBins(Base):
    """Discretize mixed features using data-driven quantile cut points."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        standardize: bool = False,
        levels: int = 3,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(n_in, n_out, standardize)
        self.levels = levels
        if rng is None:
            rng = np.random.default_rng(0)
        quantiles = np.sort(rng.random(size=levels - 1))
        self.quantiles = torch.tensor(quantiles).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bucket values into bins defined by sampled quantiles."""
        x = self.preprocess(x)[..., 0]
        thresholds = torch.quantile(x.flatten(), self.quantiles)
        x = torch.bucketize(x, thresholds)
        return x


# --- stochastic post-hoc layers
class Stochastic(Base):
    """Base post-hoc layer that injects Gaussian noise before decoding."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        standardize: bool = False,
        sigma: float = 0.01,
    ):
        super().__init__(n_in, n_out, standardize)
        self.sigma = sigma  # noise standard deviation

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Apply base preprocessing and add i.i.d. Gaussian perturbation."""
        x = super().preprocess(x)
        x = x + torch.randn_like(x) * self.sigma
        return x


class Categorical(Stochastic):
    """Sample dummy-coded categorical outputs from noisy logits.

    Produces n_out binary columns representing the non-reference levels of a
    (n_out + 1)-level categorical variable — i.e. standard dummy coding.
    Exactly one column is 1 per row (or all zeros for the reference level).
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        standardize: bool = False,
        sigma: float = 0.01,
        temperature: float = 1.0,
    ):
        super().__init__(
            n_in=n_in, n_out=n_out, standardize=standardize, sigma=sigma
        )
        self.temperature = max(float(temperature), 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sample one-hot classes with an implicit reference category."""
        logits = self.preprocess(x)[..., 0] / self.temperature

        # include reference category and get probs
        zeros = torch.zeros_like(logits[..., 0:1])
        logits = torch.cat([logits, zeros], dim=-1)
        probs = F.softmax(logits, dim=-1)

        # sample categories and dummy-code non-reference levels
        shape = probs.shape
        ids = torch.multinomial(probs.reshape(-1, shape[-1]), 1).reshape(
            shape[:-1]
        )
        out = F.one_hot(ids, num_classes=self.n_out + 1)[..., 1:].float()
        return out


class CategoricalBlock(nn.Module):
    """Dummy columns for multiple independent categorical variables in one shot.

    Models the structure of real design matrices: k categorical predictors, each
    contributing (n_levels_i - 1) binary dummy columns.  Within each group the
    columns are mutually exclusive (exactly one is 1 per row, or all zeros for
    the reference level); across groups they are independent because each
    categorical gets its own Dirichlet mixing weights.

    n_levels is a list of per-categorical level counts (≥ 2 each).
    Total output columns = sum(n_levels_i - 1).
    """

    def __init__(
        self,
        n_in: int,
        n_levels: list[int],
        standardize: bool = False,
        sigma: float = 0.01,
    ):
        super().__init__()
        self.cats = nn.ModuleList(
            [
                Categorical(
                    n_in=n_in,
                    n_out=k - 1,
                    standardize=standardize,
                    sigma=sigma,
                )
                for k in n_levels
            ]
        )

    @property
    def n_out(self) -> int:
        return sum(c.n_out for c in self.cats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate dummies for each categorical and concatenate."""
        return torch.cat([c(x) for c in self.cats], dim=-1)


class OrdinalBlock(nn.Module):
    """Generate k correlated ordinal features via shared-direction thresholding.

    Samples k projection directions clustered around a shared base direction
    (controlled by ``noise_scale``).  Each direction is passed through
    independently sampled thresholds, producing ordinal columns (integer values
    0 … L-1) that are strongly rank-correlated with each other because they
    share the same underlying latent axis — matching the Spearman > Pearson
    pattern common in real datasets with multiple ordinal predictors.

    Parameters
    ----------
    n_in:
        Number of input features.
    n_out:
        Number of ordinal output columns.
    rng:
        NumPy random generator.
    """

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator):
        super().__init__()
        self.n_out = n_out

        # shared base direction + per-feature perturbations
        base = rng.standard_normal(n_in)
        base /= np.linalg.norm(base) + 1e-8
        noise_scale = float(rng.uniform(0.05, 0.5))
        dirs = base[None, :] + rng.standard_normal((n_out, n_in)) * noise_scale
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs /= norms + 1e-8
        self.register_buffer('W', torch.from_numpy(dirs).float())  # (n_out, n_in)

        # independent thresholds per ordinal column, same number of levels
        n_levels = int(rng.integers(3, 8))
        tau = rng.standard_normal((n_out, n_levels - 1))
        tau.sort(axis=1)
        self.register_buffer('tau', torch.from_numpy(tau).float())  # (n_out, L-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.W.T                                    # (n, n_out)
        # count how many thresholds each projection exceeds
        ordinal = (z.unsqueeze(-1) > self.tau.unsqueeze(0)).float().sum(-1)
        return ordinal                                       # (n, n_out)


class Poisson(Stochastic):
    """Sample count-valued outputs using a Poisson likelihood."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Exponentiate logits into rates and draw Poisson samples."""
        x = self.preprocess(x)[..., 0]
        lam = x.exp()
        x = torch.poisson(lam)
        return x


class NegativeBinomial(Stochastic):
    """Sample overdispersed counts via a Negative Binomial model."""

    @property
    def nParam(self) -> int:
        """Use two parameters per output: logits and total count."""
        return 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split mixed features into logits and counts, then sample."""
        p, r = self.preprocess(x).split(1, dim=-1)
        p = p.squeeze(-1)
        r = F.softplus(r.squeeze(-1))
        x = D.NegativeBinomial(total_count=r, logits=p).sample()
        return x


class Clamp(Base):
    """Clip values to random quantile-derived bounds, creating floor/ceiling effects."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        standardize: bool = False,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(n_in, n_out, standardize)
        if rng is None:
            rng = np.random.default_rng(0)
        self.q_lo = float(rng.uniform(0.0, 0.15))
        self.q_hi = float(rng.uniform(0.85, 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)[..., 0]
        lo = torch.quantile(x.flatten(), self.q_lo)
        hi = torch.quantile(x.flatten(), self.q_hi)
        return x.clamp(lo, hi)


class CensoredFloor(Base):
    """Replace values below a random quantile threshold with that threshold (detection-limit censoring)."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        standardize: bool = False,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(n_in, n_out, standardize)
        if rng is None:
            rng = np.random.default_rng(0)
        self.q_floor = float(rng.uniform(0.05, 0.40))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)[..., 0]
        floor_val = torch.quantile(x.flatten(), self.q_floor)
        return x.clamp(min=floor_val)


POSTHOC_LAYERS = (
    Threshold,
    MultiThreshold,
    QuantileBins,
    Categorical,
    Poisson,
    NegativeBinomial,
    Clamp,
    CensoredFloor,
)


class Posthoc(nn.Module):
    """Apply optional post-hoc feature transformations to SCM outputs."""

    def __init__(
        self,
        n_features: int,
        p_posthoc: float = 0.2,  # probability that *at least one* feature is transformed
        rng: np.random.Generator | None = None,
    ):
        """Initialize a random set of post-hoc transformation layers.

        p_posthoc is the probability that at least one feature is transformed.
        When the dataset-level coin flip fires, the number of transformed
        features is drawn from Binomial(n_features, p_posthoc) clamped to ≥1.
        This prevents the structural under-firing that occurred for small d
        when p_posthoc was applied independently per feature.
        """
        super().__init__()
        self.n_features = n_features
        if rng is None:
            rng = np.random.default_rng(0)
        self.rng = rng

        # dataset-level flip: does any posthoc transformation happen at all?
        if p_posthoc > 0 and self.rng.random() < p_posthoc:
            self.n_posthoc = max(
                1, int(self.rng.binomial(n_features, p_posthoc))
            )
        else:
            self.n_posthoc = 0

        # build transformation layers
        layers = []

        for _ in range(self.n_posthoc):
            # Route to a block-level transform (categorical or ordinal) 60% of
            # the time, and to an individual layer 40% of the time.
            # - CategoricalBlock: multiple independent dummy-coded categoricals
            # - OrdinalBlock: multiple correlated ordinal features via shared
            #   projection directions (targets Spearman > Pearson gap)
            r = self.rng.random()
            if r < 0.30:
                n_cats = int(self.rng.integers(1, max(3, n_features // 2) + 1))
                n_levels = [
                    int(
                        self.rng.integers(
                            2, max(4, n_features // max(1, n_cats)) + 1
                        )
                    )
                    for _ in range(n_cats)
                ]
                layers.append(
                    CategoricalBlock(
                        n_in=n_features, n_levels=n_levels, standardize=True
                    )
                )
            elif r < 0.40:
                n_out = int(self.rng.integers(2, max(4, n_features // 2) + 1))
                layers.append(OrdinalBlock(n_in=n_features, n_out=n_out, rng=self.rng))
            else:
                # n_out: number of output columns for this transform.
                # Upper bound is inclusive of n_features so a single Categorical
                # can cover all d columns (was off-by-one before).
                n_out = int(self.rng.integers(1, max(4, n_features) + 1))
                # upweight skew-inducing layers (Poisson, NegBinomial, CensoredFloor)
                # to close the skewness coverage gap observed in diagnostic benchmarks
                _SKEW_LAYERS = (Poisson, NegativeBinomial, CensoredFloor)
                weights = np.array(
                    [3.0 if cls in _SKEW_LAYERS else 1.0 for cls in POSTHOC_LAYERS],
                    dtype=float,
                )
                weights /= weights.sum()
                layer_cls = self.rng.choice(POSTHOC_LAYERS, p=weights)  # type: ignore
                cfg: dict = {
                    'n_in': n_features,
                    'n_out': n_out,
                    'standardize': True,
                }
                if layer_cls in (
                    MultiThreshold,
                    QuantileBins,
                    Clamp,
                    CensoredFloor,
                ):
                    cfg['rng'] = self.rng
                layers.append(layer_cls(**cfg))

        self.transformations = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_posthoc > 0:
            out = []

            # try out each transform
            for t in self.transformations:
                h = t(x)
                if sanityCheck(h):
                    out.append(h)

            # Fix A: posthoc columns always replace SCM columns rather than
            # competing with them in a random permutation.  Cap at n_features,
            # then fill the remainder with randomly chosen SCM columns.
            if out:
                z = torch.cat(out, dim=-1)
                n_keep_posthoc = min(z.shape[-1], self.n_features)
                n_keep_scm = self.n_features - n_keep_posthoc
                scm_idx = torch.randperm(x.shape[-1])[:n_keep_scm]
                ph_idx = torch.randperm(z.shape[-1])[:n_keep_posthoc]
                x = torch.cat([x[:, scm_idx], z[:, ph_idx]], dim=-1)
        return x
