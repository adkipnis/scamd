"""Activation-pool factory for composing basic, GP, and random-choice activations."""

from functools import partial
from typing import Callable, Literal

import numpy as np

from .basic import basic_activations
from .gp import GP
from .meta import RandomScaleFactory, RandomChoiceFactory
from .utils import getRng


def getActivations(
    n_gp: int = 12,
    n_random_choice: int = 8,
    random_scale: bool = True,
    gp_types: tuple[Literal['se', 'matern', 'fractional'], ...] = (
        'se',
        'matern',
        'fractional',
    ),
    gp_type_probs: tuple[float, ...] = (0.35, 0.25, 0.40),
    n_choice: int = 1,
    allow_nested_random_choice: bool = False,
    include_basic: bool = True,
) -> list[Callable]:
    """Build activation callables used by SCM layers.

    This utility balances expressivity and stability by sampling from a base pool
    (basic activations + GP variants) and optionally adding random-choice mixers.

    Args:
        n_gp: Number of GP activation factories to add to the base pool.
        n_random_choice: Number of `RandomChoiceFactory` entries to append.
        random_scale: Wrap each base activation as
            `Standardizer -> RandomScale -> activation`.
        gp_types: Allowed GP kernel families to sample from.
        gp_type_probs: Sampling probabilities for `gp_types`.
        n_choice: Number of activation branches used inside `RandomChoice`.
        allow_nested_random_choice: If `True`, random-choice activations may
            sample other random-choice activations. If `False`, they sample only
            from the base activation pool.
        include_basic: Include handcrafted/basic activations in the base pool.
    """
    if n_gp < 0:
        raise ValueError('n_gp must be >= 0')
    if n_random_choice < 0:
        raise ValueError('n_random_choice must be >= 0')
    if n_choice <= 0:
        raise ValueError('n_choice must be > 0')
    if len(gp_types) == 0:
        raise ValueError('gp_types must be non-empty')
    if len(gp_types) != len(gp_type_probs):
        raise ValueError('gp_types and gp_type_probs must have same length')

    rng = getRng()
    probs = np.asarray(gp_type_probs, dtype=float)
    if (probs < 0).any() or probs.sum() <= 0:
        raise ValueError('gp_type_probs must be non-negative and sum to > 0')
    probs = probs / probs.sum()

    base_pool: list[Callable] = []
    if include_basic:
        base_pool.extend(basic_activations.copy())

    # Add GP factories with explicit kernel draws to control morphology mix.
    sampled_gp_types = rng.choice(gp_types, size=n_gp, p=probs)
    base_pool.extend(
        partial(GP, gp_type=str(gp_type)) for gp_type in sampled_gp_types
    )

    if random_scale:
        base_pool = [RandomScaleFactory(act) for act in base_pool]

    activations = list(base_pool)
    if n_random_choice == 0:
        return activations

    # Keep random-choice sampling shallow by default to avoid nested mixtures.
    candidate_pool = (
        activations if allow_nested_random_choice else list(base_pool)
    )
    activations.extend(
        RandomChoiceFactory(candidate_pool, n_choice=n_choice)
        for _ in range(n_random_choice)
    )
    return activations
