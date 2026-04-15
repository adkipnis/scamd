"""Sparse DAG-based structural causal model generator."""

from typing import Callable

import networkx as nx
import numpy as np
import torch
from torch import nn

from scamd.utils import hasConstantColumns, sanityCheck


def sampleDag(
    n_nodes: int,
    graph: str = 'barabasi_albert',
    m: int = 2,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, list[int]]]:
    """Return a list of (node, parents) pairs in topological order.

    Parameters
    ----------
    n_nodes:
        Total number of nodes (observed + latent).
    graph:
        ``'barabasi_albert'`` for a scale-free DAG or ``'erdos_renyi'`` for a
        random sparse DAG.
    m:
        Barabási-Albert attachment parameter (average in-degree ≈ m).  For
        Erdős-Rényi this is used as the expected in-degree.
    rng:
        NumPy random generator.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if graph == 'barabasi_albert':
        # nx.barabasi_albert_graph is undirected; orient edges toward higher
        # index nodes to get a DAG.
        g_undirected = nx.barabasi_albert_graph(
            n_nodes, m, seed=int(rng.integers(0, 2**31))
        )
        g = nx.DiGraph()
        g.add_nodes_from(range(n_nodes))
        for u, v in g_undirected.edges():
            if u < v:
                g.add_edge(u, v)
            else:
                g.add_edge(v, u)
    elif graph == 'erdos_renyi':
        p = min(m / max(n_nodes - 1, 1), 0.9)
        g = nx.erdos_renyi_graph(
            n_nodes, p, directed=True, seed=int(rng.integers(0, 2**31))
        )
        # remove backward edges to keep it acyclic
        g = nx.DiGraph((u, v) for u, v in g.edges() if u < v)
    else:
        raise ValueError(
            f"unknown graph type {graph!r}; choose 'barabasi_albert' or 'erdos_renyi'"
        )

    order = list(nx.topological_sort(g))
    return [(node, list(g.predecessors(node))) for node in order]


class DAGLayer(nn.Module):
    """One node in a DAG-SCM: x_i = activation(W @ x_parents) + noise."""

    def __init__(
        self,
        n_parents: int,
        activation: Callable,
        sigma_e: float | torch.Tensor = 0.01,
        sigma_w: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        super().__init__()
        self.n_parents = n_parents
        self.activation_factory = activation
        self.sigma_w = sigma_w
        if rng is None:
            rng = np.random.default_rng(0)
        self.rng = rng

        # weight matrix: (1, n_parents) or no weight for roots
        if n_parents > 0:
            self.linear = nn.Linear(n_parents, 1, bias=True)
        else:
            self.linear = None

        if isinstance(sigma_e, torch.Tensor):
            self.sigma_e = sigma_e
        else:
            self.sigma_e = torch.tensor(float(sigma_e))

    def _initWeights(self) -> None:
        if self.linear is not None:
            nn.init.normal_(
                self.linear.weight, std=self.sigma_w / (self.n_parents**0.5)
            )
            nn.init.zeros_(self.linear.bias)

    def forward(self, parent_values: list[torch.Tensor]) -> torch.Tensor:
        """Compute node value from parent activations.

        Parameters
        ----------
        parent_values:
            List of (n_samples,) tensors, one per parent.

        Returns
        -------
        (n_samples,) tensor.
        """
        self._initWeights()
        if self.linear is not None and parent_values:
            x_in = torch.stack(parent_values, dim=-1)  # (n, n_parents)
            h = self.linear(x_in).squeeze(-1)          # (n,)
            h = self.activation_factory()(h)
        else:
            # root node: standard normal draw
            n = parent_values[0].shape[0] if parent_values else 1
            h = torch.zeros(n)

        noise = torch.randn_like(h) * self.sigma_e
        return h + noise


class DAGSCM(nn.Module):
    """Sample synthetic features via a sparse DAG-based structural causal model.

    Parameters
    ----------
    n_observed:
        Number of observed output features.
    n_latent:
        Number of additional latent (hidden) nodes in the DAG.
    graph:
        DAG topology: ``'barabasi_albert'`` (default) or ``'erdos_renyi'``.
    m:
        Attachment parameter for the graph sampler (≈ average in-degree).
    activation:
        Activation class applied at each non-root node.
    sigma_w:
        Standard deviation for weight initialisation.
    sigma_e:
        Base additive noise scale (calibrated if ``calibrate_noise=True``).
    calibrate_noise:
        If True, scale per-node sigma_e to ``calibration_frac × IQR`` using a
        pilot forward pass with root-normal inputs.
    calibration_frac:
        Fraction of signal IQR used as noise standard deviation.
    calibration_n:
        Number of pilot samples for noise calibration.
    rng:
        NumPy random generator.
    """

    def __init__(
        self,
        n_observed: int,
        n_latent: int = 10,
        graph: str = 'barabasi_albert',
        m: int = 2,
        activation: Callable = nn.ReLU,
        sigma_w: float = 1.0,
        sigma_e: float = 0.01,
        calibrate_noise: bool = True,
        calibration_frac: float = 0.1,
        calibration_n: int = 256,
        rng: np.random.Generator | None = None,
    ):
        super().__init__()
        self.n_observed = n_observed
        self.n_latent = n_latent
        self.n_nodes = n_observed + n_latent
        self.calibrate_noise = calibrate_noise
        self.calibration_frac = calibration_frac
        self.calibration_n = calibration_n
        if rng is None:
            rng = np.random.default_rng(0)
        self.rng = rng

        # sample DAG topology
        self.topo_order = sampleDag(self.n_nodes, graph=graph, m=m, rng=rng)

        # build one DAGLayer per node
        node_layers: dict[int, DAGLayer] = {}
        for node, parents in self.topo_order:
            node_layers[node] = DAGLayer(
                n_parents=len(parents),
                activation=activation,
                sigma_e=sigma_e,
                sigma_w=sigma_w,
                rng=rng,
            )
        # store as ModuleDict (keyed by string)
        self.node_layers = nn.ModuleDict(
            {str(k): v for k, v in node_layers.items()}
        )
        self._topo_parents: dict[int, list[int]] = {
            node: parents for node, parents in self.topo_order
        }

    def _forwardOnce(self, n_samples: int) -> dict[int, torch.Tensor]:
        """Run one forward pass, returning all node activations."""
        node_vals: dict[int, torch.Tensor] = {}
        for node, parents in self.topo_order:
            layer = self.node_layers[str(node)]
            parent_vals = [node_vals[p] for p in parents]
            if not parent_vals:
                # root node: draw from standard normal
                val = torch.randn(n_samples)
            else:
                val = layer(parent_vals)
            val = torch.where(torch.isfinite(val), val, torch.zeros_like(val))
            node_vals[node] = val
        return node_vals

    def _calibrateNoise(self, n_samples: int) -> None:
        """Set per-node sigma_e = calibration_frac * IQR using a noiseless pilot."""
        # temporarily zero noise on all layers
        for layer in self.node_layers.values():
            layer.sigma_e = torch.zeros(1)

        with torch.no_grad():
            node_vals = self._forwardOnce(n_samples)

        # set calibrated sigma per node
        for node, _ in self.topo_order:
            vals = node_vals[node]
            q75, q25 = torch.quantile(vals, 0.75), torch.quantile(vals, 0.25)
            iqr = (q75 - q25).clamp(min=1e-6)
            self.node_layers[str(node)].sigma_e = (
                self.calibration_frac * iqr
            ).detach()

    def forward(self, n_samples: int) -> torch.Tensor | None:
        """Generate a (n_samples, n_observed) feature matrix.

        Unlike the MLP-based SCM, DAGSCM does not take a causes tensor — root
        nodes are drawn from N(0,1) internally so that causally earlier nodes
        are always ancestors of later ones.
        """
        if self.calibrate_noise and not getattr(
            self, '_noise_calibrated', False
        ):
            pilot_n = min(n_samples, self.calibration_n)
            self._calibrateNoise(pilot_n)
            self._noise_calibrated = True

        node_vals = self._forwardOnce(n_samples)

        # pick n_observed nodes — prefer leaf nodes (nodes with no children)
        all_nodes = [node for node, _ in self.topo_order]
        children_set: set[int] = set()
        for _, parents in self.topo_order:
            children_set.update(parents)
        leaves = [n for n in all_nodes if n not in children_set]

        if len(leaves) >= self.n_observed:
            chosen = self.rng.choice(
                leaves, size=self.n_observed, replace=False
            )
        elif len(all_nodes) >= self.n_observed:
            chosen = self.rng.choice(
                all_nodes, size=self.n_observed, replace=False
            )
        else:
            return None

        x = torch.stack([node_vals[int(n)] for n in chosen], dim=-1)

        if not sanityCheck(x):
            return None
        return x
