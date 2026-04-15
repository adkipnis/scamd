"""Sampling utilities for root causes in the SCM generator."""

import numpy as np
import torch
from torch import nn


class CauseSampler(nn.Module):
    def __init__(
        self,
        n_causes: int,
        dist: str = 'mixed',  # [mixed, normal, uniform]
        fixed_moments: bool = False,  # random parameters for dist
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.n_causes = n_causes

        # set rng
        if rng is None:
            rng = np.random.default_rng(0)
        self.rng = rng

        # set distribution
        self.dist = {
            'normal': self.normal,
            'uniform': self.uniform,
            'mixed': self.mixed,
        }[dist]
        self.fixed = fixed_moments
        if not self.fixed:
            self.mu = torch.randn(n_causes)
            self.sigma = (torch.randn(n_causes) * self.mu).abs()

    def normal(self, shape: tuple[int, int]) -> torch.Tensor:
        x = torch.randn(*shape)
        if not self.fixed:
            mu, sigma = self.mu.unsqueeze(0), self.sigma.unsqueeze(0)
            x = mu + x * sigma
        return x

    def uniform(self, shape: tuple[int, int]) -> torch.Tensor:
        x = torch.rand(*shape)
        if not self.fixed:
            mu, sigma = self.mu.unsqueeze(0), self.sigma.unsqueeze(0)
            x = mu + (x - 0.5) * sigma * np.sqrt(12)
        return x

    def _multinomial(self, shape: tuple[int, int]) -> torch.Tensor:
        n, d = shape
        n_categories = int(torch.randint(low=2, high=20, size=(1,))[0])
        probs = torch.rand((d, n_categories))
        x = torch.multinomial(probs, n, replacement=True).permute(1, 0).float()
        x = (x - x.mean(0)) / x.std(0)
        return x

    def _zipf(self, shape: tuple[int, int]) -> torch.Tensor:
        a = 2 * self.rng.random() + 2
        x = self.rng.zipf(a, shape)
        x = torch.from_numpy(x).clamp(max=10).float()
        x = (x - x.mean(0)) / x.std(0)
        return x

    def _gamma(self, shape: tuple[int, int]) -> torch.Tensor:
        k = self.rng.uniform(0.5, 5.0)   # shape parameter
        theta = self.rng.uniform(0.5, 3.0)  # scale parameter
        x = torch.from_numpy(self.rng.gamma(k, theta, shape)).float()
        x = (x - x.mean(0)) / x.std(0).clamp(min=1e-6)
        return x

    def _lognormal(self, shape: tuple[int, int]) -> torch.Tensor:
        mu = self.rng.uniform(-1.0, 1.0)
        sigma = self.rng.uniform(0.3, 1.5)
        x = torch.from_numpy(self.rng.lognormal(mu, sigma, shape)).float()
        x = (x - x.mean(0)) / x.std(0).clamp(min=1e-6)
        return x

    def _beta(self, shape: tuple[int, int]) -> torch.Tensor:
        a = self.rng.uniform(0.5, 5.0)
        b = self.rng.uniform(0.5, 5.0)
        x = torch.from_numpy(self.rng.beta(a, b, shape)).float()
        x = (x - x.mean(0)) / x.std(0).clamp(min=1e-6)
        return x

    def _studentT(self, shape: tuple[int, int]) -> torch.Tensor:
        df = self.rng.uniform(2.5, 10.0)  # keep finite variance
        x = torch.from_numpy(self.rng.standard_t(df, shape)).float()
        x = x.clamp(-10, 10)
        x = (x - x.mean(0)) / x.std(0).clamp(min=1e-6)
        return x

    def _mixtureGaussian(self, shape: tuple[int, int]) -> torch.Tensor:
        n, d = shape
        n_components = int(self.rng.integers(2, 5))
        weights = self.rng.dirichlet(np.ones(n_components))
        counts = self.rng.multinomial(n, weights)
        means = self.rng.normal(0, 2.0, size=(n_components, d))
        stds = np.abs(self.rng.normal(0.5, 0.5, size=(n_components, d))) + 0.1
        parts = []
        for i in range(n_components):
            z = self.rng.normal(means[i], stds[i], size=(counts[i], d))
            parts.append(torch.from_numpy(z).float())
        x = torch.cat(parts, dim=0)
        # shuffle rows so components are interleaved
        x = x[torch.randperm(n)]
        x = (x - x.mean(0)) / x.std(0).clamp(min=1e-6)
        return x

    def mixed(self, shape: tuple[int, int]) -> torch.Tensor:
        out = []
        dists = [
            torch.randn,
            torch.rand,
            self._multinomial,
            self._zipf,
            self._gamma,
            self._lognormal,
            self._beta,
            self._studentT,
            self._mixtureGaussian,
        ]
        n, d = shape

        # draw distributions
        probs = self.rng.dirichlet(alpha=np.ones((len(dists),)), size=(d,))
        ids = np.sort(probs.argmax(-1))
        ids, counts = np.unique_counts(ids)

        # draw from each distribution
        for idx, d_ in zip(ids, counts):
            dist = dists[idx]
            x = dist((n, d_))
            out.append(x)

        # gather and permute positions
        x = torch.cat(out, dim=-1)
        x = x[:, torch.randperm(d)]

        # optionally rescale
        if not self.fixed:
            mu, sigma = self.mu.unsqueeze(0), self.sigma.unsqueeze(0)
            x = mu + x * sigma
        return x

    def sample(self, n_samples: int) -> torch.Tensor:
        shape = (n_samples, self.n_causes)
        return self.dist(shape)
