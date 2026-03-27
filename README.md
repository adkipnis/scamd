# Structural CAusal Model Datsets (scamd)

Generate realistic synthetic tabular datasets with a structural causal model (SCM).

`scm-datasets` builds latent causes, transforms them through deep nonlinear mechanisms, and applies optional post-hoc feature transformations (categorical/count/rank/binning) to create mixed-type, high-dependency feature matrices.

## Design at a glance

1. **Cause sampling** (`scamd/causes.py`)
   Sample root variables from configurable distributions.

2. **Structural mechanism** (`scamd/scm.py`)
   Pass causes through a deep noisy MLP with stochastic activations/weights.

3. **Post-hoc feature transforms** (`scamd/posthoc.py`)
   Inject realistic tabular effects (thresholding, quantiles, categorical/count outputs).

## Install

```bash
uv sync
```

Python `>=3.12` is required.

## Quickstart

```python
from torch import nn

from scamd import generate_dataset
from scamd.utils import setSeed

setSeed(42)

x = generate_dataset(
    n_samples=1000,
    n_features=20,
    n_causes=12,
    cause_dist="mixed",   # "normal" | "uniform" | "mixed"
    n_layers=8,
    n_hidden=64,
    activation=nn.Tanh,
    blockwise=True,
)
```

See `examples/quickstart.py` for a runnable script.
