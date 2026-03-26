"""Sampling utilities for root causes in the SCM generator."""

import numpy as np
import torch
from torch import nn
from typing import Iterable

from .utils import getRng


class CauseSampler(nn.Module):
    def __init__(
        self,
        n_samples: int,
        n_causes: int,
        dist: str = 'normal',  # [mixed, normal, uniform]
        fixed: bool = False,  # random parameters for dist
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.n_causes = n_causes
        self.dist = {
            'normal': self.normal,
            'uniform': self.uniform,
            'mixed': self.mixed,
        }[dist]
        self.fixed = fixed
        if not self.fixed:
            self.mu = torch.randn(n_causes)
            self.sigma = (torch.randn(n_causes) * self.mu).abs()

    def normal(self, shape: Iterable) -> torch.Tensor:
        x = torch.randn(*shape)
        if not self.fixed:
            mu, sigma = self.mu.unsqueeze(0), self.sigma.unsqueeze(0)
            x = mu + x * sigma
        return x

    def uniform(self, shape: Iterable) -> torch.Tensor:
        x = torch.rand(*shape)
        if not self.fixed:
            mu, sigma = self.mu.unsqueeze(0), self.sigma.unsqueeze(0)
            x = mu + (x - 0.5) * sigma * np.sqrt(12)
        return x

    def _multinomial(self, shape: Iterable) -> torch.Tensor:
        n, d = shape
        n_categories = getRng().integers(2, 20)
        probs = torch.rand((d, n_categories))
        x = torch.multinomial(probs, n, replacement=True).permute(1, 0).float()
        x = (x - x.mean(0)) / x.std(0)
        return x

    def _zipf(self, shape: Iterable) -> torch.Tensor:
        a = 2 * getRng().random() + 2
        x = getRng().zipf(a, shape)  # type: ignore
        x = torch.tensor(x).clamp(max=10).float()
        x = (x - x.mean(0)) / x.std(0)
        return x

    def mixed(self, shape: Iterable) -> torch.Tensor:
        out = []
        dists = [torch.randn, torch.rand, self._multinomial, self._zipf]
        n, d = shape

        # draw distributions
        probs = getRng().dirichlet(alpha=np.ones((4,)), size=(d,))
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

    def sample(self) -> torch.Tensor:
        shape = (self.n_samples, self.n_causes)
        return self.dist(shape)
