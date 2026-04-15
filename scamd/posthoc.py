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
            D.Dirichlet(alpha).sample((n_out, self.n_param)).permute(2, 0, 1)
        )

    @property
    def n_param(self) -> int:
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
    def n_param(self) -> int:
        """Use two parameters per output: logits and total count."""
        return 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split mixed features into logits and counts, then sample."""
        p, r = self.preprocess(x).split(1, dim=-1)
        p = p.squeeze(-1)
        r = F.softplus(r.squeeze(-1))
        x = D.NegativeBinomial(total_count=r, logits=p).sample()
        return x


POSTHOC_LAYERS = (
    Threshold,
    MultiThreshold,
    QuantileBins,
    Categorical,
    Poisson,
    NegativeBinomial,
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
            # Fix C: wider range of dummy levels — up to max(4, n_features) levels
            # so multi-level categoricals with more dummies can be represented.
            n_out = int(self.rng.integers(1, max(4, n_features)))
            layer_cls = self.rng.choice(POSTHOC_LAYERS)  # type: ignore
            # Fix D: MultiThreshold and QuantileBins accept rng; others don't
            cfg: dict = {
                'n_in': n_features,
                'n_out': n_out,
                'standardize': True,
            }
            if layer_cls in (MultiThreshold, QuantileBins):
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

            # append them to the original and take a subset of both
            if out:
                z = torch.cat(out, dim=-1)
                x = torch.cat([x, z], dim=-1)
                idx = torch.randperm(x.shape[-1])[: self.n_features]
                x = x[..., idx]
        return x
