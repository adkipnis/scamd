"""Activation factory for composing basic, GP, and random-choice activations."""

from .basic import basic_activations
from .gp import GP
from .meta import RandomScaleFactory, RandomChoiceFactory


def getActivations() -> list:
    activations = basic_activations.copy()
    activations += [GP] * 12
    activations = [RandomScaleFactory(act) for act in activations]
    # ks = [2 ** logUniform(0.1, 4, round=True)
    #       for _ in range(len(activations))]
    # out = [RandomChoiceFactory(activations, int(k)) for k in ks]
    activations += [RandomChoiceFactory(activations)] * 12
    return activations
