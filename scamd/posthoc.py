"""Deterministic and stochastic post-hoc feature transformations."""

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D
import numpy as np
from .meta import Standardizer
from .utils import getRng


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
        self, n_in: int, n_out: int, standardize: bool = False, levels: int = 3
    ):
        super().__init__(n_in, n_out, standardize)
        self.levels = levels  # number of thresholds
        self.tau = np.sort(getRng().normal(size=levels - 1))

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
        self, n_in: int, n_out: int, standardize: bool = False, levels: int = 3
    ):
        super().__init__(n_in, n_out, standardize)
        self.levels = levels  # number of quantiles
        quantiles = np.sort(getRng().random(size=levels - 1))
        self.quantiles = torch.tensor(quantiles).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bucket values into bins defined by sampled quantiles."""
        x = self.preprocess(x)[..., 0]
        thresholds = torch.quantile(x.flatten(), self.quantiles)
        x = torch.bucketize(x, thresholds)
        return x


class Rank(Base):
    """Return indices of the top-k features per sample."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute top-k feature indices with k capped at input width."""
        k = min(self.n_out, x.shape[-1])
        x = torch.topk(x, k).indices
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
    """Produce one-hot categorical outputs from noisy logits."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert logits to one-hot classes with an implicit reference class."""
        x = self.preprocess(x)[..., 0]

        # include reference category and get probs
        zeros = torch.zeros_like(x[..., 0:1])
        x = torch.cat([x, zeros], dim=-1)
        probs = F.softmax(x, dim=-1)

        # dummy code categories
        x = probs.argmax(-1)
        x = F.one_hot(x, num_classes=self.n_out + 1)[..., 1:]
        return x


class Poisson(Stochastic):
    """Sample count-valued outputs using a Poisson likelihood."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Exponentiate logits into rates and draw Poisson samples."""
        x = self.preprocess(x)[..., 0]
        lam = x.exp()
        x = torch.poisson(lam)
        return x


class Geometric(Stochastic):
    """Sample geometric random variables parameterized by logits."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw geometric samples from the transformed logits."""
        x = self.preprocess(x)[..., 0]
        x = D.Geometric(logits=x).sample()
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


def getPosthocLayers() -> list[nn.Module]:
    """Return all available post-hoc layer classes."""
    deterministic = [Threshold, MultiThreshold, QuantileBins, Rank]
    stochastic = [Categorical, Poisson, Geometric, NegativeBinomial]
    return deterministic + stochastic
