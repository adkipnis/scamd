"""Minimal end-to-end example: generate and visualise a SCAMD dataset."""

import pandas as pd
from matplotlib import pyplot as plt

from scamd import generate_dataset, plot_dataset

x = generate_dataset(
    n_samples=300,
    n_features=8,
    n_causes=12,
    n_layers=8,
    n_hidden=16,
    blockwise=False,
    preset='balanced_realistic',
)

df = pd.DataFrame(x[:, :4], columns=[f'x{i + 1}' for i in range(4)])
plot_dataset(df, color='teal', title='scamd quickstart sample', kde=False)
plt.show()
