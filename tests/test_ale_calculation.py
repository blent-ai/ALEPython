# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from alepython.ale import _first_order_ale_quant, _get_centres, _second_order_ale_quant

from .utils import interaction_predictor, linear_predictor


def test_linear():
    """We expect both X['a'] and X['b'] to have a linear relationship.

    There should be no second order interaction.

    """

    def linear(x, a, b):
        return a * x + b

    np.random.seed(1)

    N = int(1e5)
    X = pd.DataFrame({"a": np.random.random(N), "b": np.random.random(N)})

    # Test that the first order relationships are linear.
    for column in X.columns:
        ale, quantiles = _first_order_ale_quant(linear_predictor, X, column, 21)
        centres = _get_centres(quantiles)
        p, V = np.polyfit(centres, ale, 1, cov=True)
        assert np.all(np.isclose(p, [1, -0.5], atol=1e-3))
        assert np.all(np.isclose(np.sqrt(np.diag(V)), 0))

    # Test that a second order relationship does not exist.
    ale_second_order, quantiles_list = _second_order_ale_quant(
        linear_predictor, X, X.columns, 21
    )
    assert np.all(np.isclose(ale_second_order, 0))


def test_interaction():
    """Ensure that the method picks up a trivial interaction term."""
    np.random.seed(1)

    N = int(1e6)
    X = pd.DataFrame({"a": np.random.random(N), "b": np.random.random(N)})
    ale, quantiles_list = _second_order_ale_quant(
        interaction_predictor, X, X.columns, 61
    )

    # XXX: There seems to be a small deviation proportional to the first axis ('a')
    # that is preventing this from being closer to 0.
    assert np.all(np.abs(ale[:, :30] - ale[:, 31:][::-1]) < 1e-2)
