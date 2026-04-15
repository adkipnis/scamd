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
        p_corr_causes: float = 0.5,  # probability of applying a Gaussian copula
        max_corr_strength: float = 0.9,  # upper bound for copula shrinkage weight
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.n_causes = n_causes
        self.p_corr_causes = p_corr_causes
        self.max_corr_strength = max_corr_strength

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

    def _applyCopula(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a Gaussian copula to induce inter-cause correlations.

        Preserves each column's marginal distribution exactly while coupling
        the columns through a randomly sampled correlation matrix R.  R is
        constructed as a random Wishart-like matrix shrunk toward the identity
        by a factor ``rho ~ Uniform(0, max_corr_strength)``.
        """
        n, k = x.shape
        if n < 10:
            return x

        # sample a random correlation matrix shrunk toward identity
        rho = float(self.rng.uniform(0.0, self.max_corr_strength))
        A = torch.from_numpy(self.rng.standard_normal((k, k))).float()
        M = A @ A.T
        d_sqrt = M.diag().clamp(min=1e-8).sqrt()
        R_raw = M / d_sqrt.unsqueeze(0) / d_sqrt.unsqueeze(1)
        R = (1.0 - rho) * torch.eye(k) + rho * R_raw
        # clip eigenvalues to ensure positive definiteness
        eigvals, eigvecs = torch.linalg.eigh(R)
        eigvals = eigvals.clamp(min=1e-6)
        R = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        d_sqrt2 = R.diag().clamp(min=1e-8).sqrt()
        R = R / d_sqrt2.unsqueeze(0) / d_sqrt2.unsqueeze(1)

        try:
            L = torch.linalg.cholesky(R)
        except Exception:
            return x

        # empirical ranks → uniform (continuity-corrected)
        ranks = x.argsort(dim=0).argsort(dim=0).float()
        u = (ranks + 0.5) / n
        u = u.clamp(1e-6, 1.0 - 1e-6)

        # uniform → standard normal via inverse error function
        z = torch.erfinv(2.0 * u - 1.0) * (2.0 ** 0.5)

        # apply Cholesky factor to induce the target correlation
        z_corr = z @ L.T

        # standard normal → uniform via normal CDF
        u_corr = 0.5 * (1.0 + torch.erf(z_corr / (2.0 ** 0.5)))
        u_corr = u_corr.clamp(1e-6, 1.0 - 1e-6)

        # uniform → original marginals via empirical quantile
        x_sorted = x.sort(dim=0).values
        idx = (u_corr * n).long().clamp(0, n - 1)
        return x_sorted.gather(0, idx)

    def sample(self, n_samples: int) -> torch.Tensor:
        shape = (n_samples, self.n_causes)
        x = self.dist(shape)
        if (
            self.p_corr_causes > 0
            and self.n_causes >= 2
            and self.rng.random() < self.p_corr_causes
        ):
            x = self._applyCopula(x)
        return x
