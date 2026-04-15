# SCAMD — Structural Causal Model Datasets

Generate realistic synthetic tabular feature matrices using structural causal models.

`scamd` is designed for researchers and practitioners who need synthetic datasets that reproduce the statistical properties of real-world design matrices: mixed feature types, non-linear dependencies, heterogeneous marginals, and discrete/categorical structure. It is particularly useful for benchmarking regression estimators, variable selection methods, and meta-learning systems that train across many datasets.

---

## Install

```bash
uv sync
uv pip install -e .
```

Python ≥ 3.12 is required. Plotting requires `pandas` and `seaborn`.

---

## Quickstart

```python
import numpy as np
from scamd import generateDataset

rng = np.random.default_rng(42)

X = generateDataset(
    n_samples=500,
    n_features=10,
    n_causes=15,
    n_layers=6,
    n_hidden=32,
    blockwise=True,
    preset='balanced_realistic',
    rng=rng,
)
# X.shape == (500, 10)
```

For a pair-grid plot of the first few features:

```python
import pandas as pd
from matplotlib import pyplot as plt
from scamd import plotDataset

df = pd.DataFrame(X[:, :6], columns=[f'x{i+1}' for i in range(6)])
plotDataset(df, color='teal', title='SCAMD sample', kde=True)
plt.show()
```

See `examples/quickstart.py` for a runnable script, and `examples/` for demos of each subsystem.

---

## Generation pipeline

Each call to `generateDataset` (or `Generator.sample`) runs three stages in sequence:

```
CauseSampler  →  SCM  →  Posthoc
   (latent         (deep        (feature
    inputs)         MLP)      transforms)
```

### Stage 1 — Root causes (`causes.py`)

The pipeline starts by sampling a matrix of *latent causes* with shape `(n_samples, n_causes)`. These are the exogenous root variables of the causal graph. Three marginal distributions are supported, selected via `cause_dist`:

| `cause_dist` | Description |
|---|---|
| `'normal'` | Independent Gaussians, optionally with per-column mean and std drawn from N(0,1) |
| `'uniform'` | Independent uniform draws, optionally shifted and scaled |
| `'mixed'` | Each column is assigned to one of nine families via a Dirichlet draw; columns are randomly permuted |

The `'mixed'` mode draws from: Gaussian, uniform, multinomial, Zipf, **gamma** (random shape/scale — right-skewed), **log-normal** (random μ/σ — heavy right tail), **beta** (random a/b — bounded [0,1]), **Student-t** (random df ≥ 2.5 — heavy symmetric tails), and **mixture-Gaussian** (2–4 components with random weights/means — multimodal). All families are standardised to zero mean before the optional rescaling step.

The `fixed=True` flag standardises all causes to zero mean and unit variance, making the SCM invariant to cause scale.

### Stage 2 — Structural causal mechanism (`scm.py`, `dag.py`, `pool.py`, `gp.py`, `meta.py`, `basic.py`)

The causes are passed through a randomly initialised structural causal model. Two implementations are available: the default **MLP** (`SCM`) and an optional **sparse DAG** (`DAGSCM`). Both are untrained — weights and topology are sampled fresh for every dataset.

#### MLP architecture (`SCM`)

The network has `n_layers` blocks. Each block is:

```
Linear(in → n_hidden)  →  NoiseLayer(σ_e)  →  Activation()
```

The first block takes `n_causes` inputs; subsequent blocks take `n_hidden`. The hidden width is automatically widened to at least `2 × n_features` to ensure there are enough distinct units to draw from. A small amount of independent Gaussian noise (`σ_e`, optionally varied per unit) is injected after each linear transform to prevent exact rank collapse.

#### Noise calibration

When `calibrate_noise=True` (the default for all presets), the MLP runs a noiseless pilot forward pass on the first call and measures the interquartile range (IQR) of each hidden unit across the pilot batch. Each `NoiseLayer`'s `σ_e` is then set to `calibration_frac × IQR` for that unit. This ensures additive noise is always proportional to signal magnitude regardless of network depth or activation scale — without it, noise can be negligible at early layers and overwhelming at later ones.

#### Sparse DAG architecture (`DAGSCM`)

The `DAGSCM` class replaces the dense MLP with a node-by-node forward pass over a randomly sampled directed acyclic graph. This produces sparser, more structured correlation patterns: each feature depends only on its direct causal parents, which eliminates the spurious correlations introduced when all features share a common latent path.

```python
from scamd import Generator
import numpy as np

gen = Generator.fromPreset(
    n_features=10, n_causes=5, n_layers=3, n_hidden=16,
    blockwise=False, preset='balanced_realistic',
    use_dag=True,                  # switch to DAGSCM
    rng=np.random.default_rng(0),
)
X = gen.sample(300)
```

Two graph topologies are supported:

| `graph` | Description |
|---|---|
| `'barabasi_albert'` (default) | Scale-free DAG: a few hub nodes influence many features; most have 1–2 parents. Matches the heterogeneous dependency structure of real-world tabular data. |
| `'erdos_renyi'` | Random sparse DAG with uniform expected in-degree. |

Edges are oriented by node index to guarantee acyclicity. The `m` parameter controls the expected in-degree (default 2). Root nodes (no parents) are drawn from N(0,1) directly; non-root nodes compute `activation(W @ x_parents) + noise`. Observed features are read out from leaf nodes where possible.

`DAGSCM` also supports noise calibration (`calibrate_noise=True`) via the same IQR-scaling approach as `SCM`.

#### Weight initialisation

Weights are re-sampled on **every forward pass**, so each call to `Generator.sample` is a new structural mechanism. Two initialisation strategies control the dependency structure:

- **Standard dropout** (`blockwise=False`): weights are drawn from N(0, σ_w²) and then multiplied by a Bernoulli(1−p) mask. This creates sparse but unstructured connectivity, producing loosely coupled features.

- **Blockwise dropout** (`blockwise=True`): the weight matrix is partitioned into a random number of block-diagonal sub-matrices (1 to √min(rows, cols) blocks). Each block receives dense Gaussian weights; the rest stays zero. This creates groups of strongly correlated features — a pattern common in real datasets where predictors cluster by theme (e.g., several items from the same survey scale).

#### Activation pool

The nonlinearity applied to each block is drawn from a large, heterogeneous pool built by `getActivations`. The pool combines three families:

**Basic activations** — 25 fixed functions including standard choices (ReLU, Tanh, Sigmoid, ELU, SELU, SiLU) and irregular ones (Abs, Square, SqrtAbs, Exp, LogAbs, Gaussian SE, Sine, Cos, Mod, Sign, Ceil, Floor, Round, UnitInterval). These cover a wide range of marginal shapes.

**GP activations** — Smooth random functions built from random Fourier features, approximating draws from a Gaussian process. Three kernel families are used:

- *Squared-exponential (SE)*: frequencies drawn from N(0, 1/ℓ²). Produces infinitely differentiable smooth functions. Length-scale ℓ is drawn from a log-uniform prior over [0.1, 16].
- *Matérn*: frequencies drawn from a Student-t distribution with fractional degrees of freedom (ν ∈ {0.5, 1.5, 2.5}). Produces functions with finite differentiability; ν = 0.5 gives rough, ν = 2.5 gives nearly smooth.
- *Fractional*: frequencies with power-law spectral decay. Scale-free, producing functions with long-range structure and self-similar roughness.

Each GP instance is independently parameterised (kernel type, length-scale, random phases, spectral weights), so the pool contains `n_gp` distinct smooth nonlinearities.

**Random-choice mixers** — `RandomChoice` layers apply several different activations to different feature subsets within a single layer, creating heterogeneous transformations across columns.

When `random_scale=True` (the default for realistic presets), each activation is wrapped as `Standardizer → RandomScale → activation`, where `RandomScale` applies a random log-normal affine transform. This broadens the scale distribution of intermediate representations.

#### Feature extraction

All layer outputs are concatenated into a matrix of shape `(n_samples, n_layers × n_hidden)`. The final `n_features` columns are selected from among non-constant units by either:

- **Random selection** (`contiguous=False`): `n_features` units drawn uniformly at random. Features may come from any layer, creating diverse marginal shapes.
- **Contiguous selection** (`contiguous=True`): a contiguous window of `n_features` adjacent units in the concatenated output, preserving more local structure from a single depth.

If a forward pass produces fewer than `n_features` non-constant units, the SCM retries up to `max_retries` times.

#### Shared noise groups

`SCM` optionally injects *shared* latent noise: a single `z ~ N(0,1)` vector is drawn per sample and added (with random weight `α`) to a random subset of features. This creates structured residual correlation between features that is not explained by the causal MLP — mimicking batch effects or latent confounders. The probability of any shared noise firing is controlled by `p_shared_noise` (default 0.5); when it fires, 1–3 independent groups are created.

### Stage 3 — Post-hoc feature transforms (`posthoc.py`)

Real design matrices contain discrete and categorical features — binary indicators, ordinal ratings, dummy-coded factors, count outcomes — that an MLP cannot produce directly. The post-hoc stage replaces a subset of continuous SCM columns with realistically structured discrete ones.

#### When transforms fire

A dataset-level coin flip (probability `p_posthoc`) decides whether *any* transformation happens at all. If it fires, the number of transform layers is drawn from Binomial(`n_features`, `p_posthoc`), clamped to at least 1. This two-stage scheme ensures that small datasets are not systematically under-transformed.

#### Mixing weights

Every post-hoc layer first **mixes** the SCM features: it draws a weight matrix from a Dirichlet distribution (ensuring the weights sum to 1 per output) and computes a weighted combination of input features. This means each output column depends on a blend of the latent SCM features, not a single one. Optionally, inputs are standardised before mixing.

#### Deterministic transforms

| Class | Output |
|---|---|
| `Threshold` | Binarises each mixed feature at zero → binary column |
| `MultiThreshold` | Counts how many of `levels−1` Gaussian-sampled thresholds a value exceeds → ordinal integer in [0, levels−1] |
| `QuantileBins` | Assigns each value to a bin defined by `levels−1` data-driven quantile cut points → discrete integer |
| `Clamp` | Clips values to random quantile-derived lower/upper bounds → continuous column with hard floor and ceiling effects |
| `CensoredFloor` | Sets values below a random quantile threshold to exactly that threshold → left-censored continuous column, mimicking instrument detection limits |

#### Stochastic transforms

| Class | Output |
|---|---|
| `Categorical` | Softmax over `n_out+1` logits (the last being a fixed reference), then multinomial sample → `n_out` binary dummy columns with at most one active per row |
| `CategoricalBlock` | Multiple independent `Categorical` layers concatenated → simulates several factor predictors each contributing their own mutually exclusive dummy group |
| `Poisson` | Exponentiates logits into rates, samples from Poisson → non-negative integer counts |
| `NegativeBinomial` | Splits mixed features into logits and (softplus) total count, samples from NegBin → overdispersed counts |

`Categorical` and `CategoricalBlock` directly model the dummy-variable structure of real design matrices. Within a group, columns are mutually exclusive (exactly one is 1 per row, or all zeros for the reference level). `CategoricalBlock` generates multiple such groups independently, matching datasets that contain several factor predictors.

#### Applying transforms

After all transform layers run, their outputs are concatenated. A random subset of up to `n_features` post-hoc columns then *replaces* SCM columns in the final output, with the remainder filled by randomly selected original SCM columns. Post-hoc columns always take priority, ensuring that the discrete structure is never crowded out.

---

## Presets

Three named presets bundle sensible defaults for the activation pool and generation parameters. Override any individual setting by passing it as a keyword argument.

| Preset | Activation mix | `p_posthoc` | Causes | Notes |
|---|---|---|---|---|
| `smooth_stable` | SE + Matern GPs only, no random-choice | 0.10 | `uniform`, fixed moments | Smooth, low-variability functions; good for testing regression estimators that assume smooth signal |
| `balanced_realistic` | All kernels, 3 random-choice layers | 0.35 | `mixed`, variable moments | Broad mix of smooth and rough functions with realistic discrete structure; the default |
| `high_variability` | Fractional-heavy, 5 random-choice layers | 0.20 | `mixed`, variable moments | Rough, scale-free nonlinearities; stress-tests estimators on irregular signal |

All three presets enable `calibrate_noise=True`.

```python
from scamd import generateDataset
import numpy as np

X = generateDataset(
    n_samples=300,
    n_features=8,
    n_causes=12,
    n_layers=5,
    n_hidden=24,
    blockwise=True,
    preset='balanced_realistic',
    rng=np.random.default_rng(0),
)
```

---

## API reference

### `generateDataset(**kwargs) → np.ndarray`

Convenience function. Returns `(n_samples, n_features)` float64 array. All arguments except `rng` are required unless a preset supplies a default.

| Argument | Type | Description |
|---|---|---|
| `n_samples` | `int` | Number of rows |
| `n_features` | `int` | Number of output columns |
| `n_causes` | `int` | Width of the latent cause layer (should be ≥ `n_features`) |
| `n_layers` | `int` | Depth of the MLP |
| `n_hidden` | `int` | Hidden units per layer (auto-widened to ≥ 2×`n_features`) |
| `blockwise` | `bool` | Use blockwise weight initialisation |
| `contiguous` | `bool` | Extract contiguous feature window (default `False`) |
| `preset` | `str` | Named preset: `'balanced_realistic'`, `'smooth_stable'`, `'high_variability'` |
| `activation` | callable or `None` | Override the sampled activation (e.g. `nn.Tanh`) |
| `cause_dist` | `str` or `None` | Root cause distribution: `'normal'`, `'uniform'`, `'mixed'` |
| `fixed` | `bool` or `None` | Fix cause moments to zero mean / unit variance |
| `p_posthoc` | `float` or `None` | Probability of any post-hoc transformation firing |
| `use_dag` | `bool` | Use sparse DAG generator (`DAGSCM`) instead of MLP (`SCM`). Default `False`. |
| `rng` | `np.random.Generator` or `None` | Explicit RNG for reproducibility |

Additional keyword arguments are forwarded to `SCM` (e.g. `calibrate_noise`, `calibration_frac`, `p_shared_noise`) or `DAGSCM` (e.g. `n_latent`, `graph`, `m`).

### `Generator`

The stateful class underlying `generateDataset`. Construct via `Generator.fromPreset(...)` for fine-grained control, or directly via `Generator(causes_config, scm_config, posthoc_config)`. Call `.sample(n_samples)`  to draw datasets.

### `plotDataset(x, ...)`

Seaborn pair-grid with histograms on the diagonal, scatter above, and optional KDE below. Accepts a NumPy array or a pandas DataFrame.

---

## Demos

```bash
python examples/quickstart.py          # generate and visualise one dataset
python examples/pool_demo.py           # activation pool presets and sampled curves
python examples/scm_demo.py            # dependency-spectrum walk-through
python examples/posthoc_demo.py        # behaviour of each post-hoc transform
python examples/causes_demo.py         # root cause distribution families
python examples/meta_demo.py           # Standardizer / RandomScale / RandomChoice
```
