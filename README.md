# SCAMD — Structural Causal Model Datasets

Generate realistic synthetic tabular feature matrices for benchmarking regression estimators, variable selection methods, and meta-learning systems.

---

## Install

```bash
uv sync && uv pip install -e .
```

Python ≥ 3.12. Plotting requires `pandas` and `seaborn`.

---

## Quickstart

```python
import numpy as np
from scamd import generateDataset

X = generateDataset(
    n_samples=500, n_features=10, n_causes=15,
    n_layers=6, n_hidden=32, blockwise=True,
    preset='balanced_realistic',
    rng=np.random.default_rng(42),
)  # X.shape == (500, 10)
```

```python
import pandas as pd
from scamd import plotDataset

df = pd.DataFrame(X[:, :6], columns=[f'x{i+1}' for i in range(6)])
plotDataset(df, color='teal', title='SCAMD sample', kde=True)
```

---

## Pipeline

```
CauseSampler  →  SCM / DAGSCM  →  Posthoc
```

### `causes.py` — Root cause sampler

Samples an `(n_samples, n_causes)` matrix of exogenous latent inputs. Three modes via `cause_dist`:

- `'normal'` / `'uniform'` — independent columns, optionally with random per-column moments
- `'mixed'` — each column is assigned to one of nine families via a Dirichlet draw: Gaussian, uniform, multinomial, Zipf, gamma, log-normal, beta, Student-t, mixture-Gaussian. Produces heterogeneous marginals including right-skewed, heavy-tailed, bounded, and multimodal columns.

### `scm.py` — MLP-based SCM

Default generator. An `n_layers`-deep network of `Linear → NoiseLayer → Activation` blocks with weights re-sampled on every forward pass, so each `sample()` call is a distinct structural mechanism. Key options:

- **`blockwise`** — block-diagonal weight matrices create clusters of correlated features; standard Bernoulli dropout creates looser coupling
- **`calibrate_noise`** — pilot forward pass scales each `NoiseLayer`'s σ to `calibration_frac × IQR`, keeping noise proportional to signal at every depth (default `True`)
- **`p_shared_noise`** — with probability 0.5, adds 1–3 shared latent noise groups that inject structured residual correlation across random feature subsets

All layer outputs are concatenated and `n_features` non-constant units are read out (randomly or contiguously).

### `dag.py` — Sparse DAG-based SCM

Optional alternative (`use_dag=True`). Generates a random DAG and evaluates nodes in topological order: root nodes draw from N(0,1); non-root nodes compute `activation(W @ x_parents) + noise`. Produces sparser, more structured correlations than the dense MLP.

- **`graph`** — `'barabasi_albert'` (default, scale-free) or `'erdos_renyi'` (uniform sparse)
- **`m`** — expected in-degree (default 2)
- Supports the same IQR noise calibration as `SCM`

### `pool.py` / `gp.py` / `basic.py` / `meta.py` — Activation pool

`getActivations()` builds the pool of nonlinearities drawn by the SCM:

- **`basic.py`** — 25 fixed activations: standard (ReLU, Tanh, SiLU, …) and irregular (Abs, Sine, Mod, Ceil, Sign, …)
- **`gp.py`** — random Fourier feature approximations to GP draws; three kernels: SE (smooth), Matérn (tunable roughness), Fractional (scale-free)
- **`meta.py`** — `RandomChoice` mixes multiple activations across feature subsets; `RandomScaleFactory` wraps any activation as `Standardizer → RandomScale → activation`

### `posthoc.py` — Post-hoc feature transforms

Replaces a random subset of continuous SCM columns with structured discrete or bounded ones. Fires with probability `p_posthoc`; when active draws the number of transforms from Binomial(`n_features`, `p_posthoc`). Each transform mixes SCM features via Dirichlet weights before applying its mapping.

| Class | Output |
|---|---|
| `Threshold` | Binary at zero |
| `MultiThreshold` | Ordinal integer via Gaussian-sampled thresholds |
| `QuantileBins` | Discrete bins via data-driven quantile cut points |
| `Clamp` | Continuous with random quantile floor/ceiling |
| `CensoredFloor` | Left-censored continuous (detection-limit effect) |
| `Categorical` / `CategoricalBlock` | Dummy-coded factor(s), mutually exclusive within group |
| `Poisson` / `NegativeBinomial` | Integer counts |

---

## Presets

| Preset | Activation mix | `p_posthoc` | Causes |
|---|---|---|---|
| `smooth_stable` | SE + Matérn GPs, no random-choice | 0.10 | `uniform`, fixed moments |
| `balanced_realistic` | All kernels, 3 random-choice layers | 0.35 | `mixed`, variable moments |
| `high_variability` | Fractional-heavy, 5 random-choice layers | 0.20 | `mixed`, variable moments |

All presets enable `calibrate_noise=True`. Any parameter can be overridden as a keyword argument.

---

## API

**`generateDataset(...) → np.ndarray`** — convenience wrapper; returns `(n_samples, n_features)` array.

Key arguments: `n_samples`, `n_features`, `n_causes`, `n_layers`, `n_hidden`, `blockwise`, `preset`, `cause_dist`, `p_posthoc`, `use_dag`, `rng`. Extra kwargs are forwarded to `SCM` (`calibrate_noise`, `calibration_frac`, `p_shared_noise`) or `DAGSCM` (`n_latent`, `graph`, `m`).

**`Generator.fromPreset(...)`** — stateful class for repeated sampling from the same mechanism. Call `.sample(n_samples)`.

**`plotDataset(x, ...)`** — Seaborn pair-grid; histograms on diagonal, scatter above, optional KDE below.

---

## Demos

```bash
python examples/quickstart.py     # generate and visualise one dataset
python examples/pool_demo.py      # activation pool curves
python examples/scm_demo.py       # dependency-spectrum walk-through
python examples/posthoc_demo.py   # post-hoc transform behaviour
python examples/causes_demo.py    # root cause distribution families
python examples/meta_demo.py      # Standardizer / RandomScale / RandomChoice
```
