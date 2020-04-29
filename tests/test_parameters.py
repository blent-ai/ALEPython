# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from alepython import ale_plot
from alepython.ale import (
    _ax_quantiles,
    _check_two_ints,
    _get_quantiles,
    _second_order_ale_quant,
)

from .utils import SimpleModel


def test_two_ints():
    with pytest.raises(ValueError, match=r"'3' values.*"):
        _check_two_ints((1, 2, 3))
    with pytest.raises(ValueError, match=r".*Got type(s) '{<class 'str'>}'.*"):
        _check_two_ints(("1", "2"))


def test_quantiles():
    with pytest.raises(ValueError, match=r".*type '<class 'str'>'.*"):
        _get_quantiles(pd.DataFrame({"a": [1, 2, 3]}), "a", "1")


def test_second_order_ale_quant():
    with pytest.raises(ValueError, match=r".*contained '1' features.*"):
        _second_order_ale_quant(lambda x: None, pd.DataFrame({"a": [1, 2, 3]}), "a", 1)


def test_ale_plot():
    """Test that proper errors are raised."""
    with pytest.raises(ValueError, match=r".*'model'.*'predictor'.*"):
        ale_plot(model=None, train_set=pd.DataFrame([1]), features=[0])

    with pytest.raises(ValueError, match=r"'3' 'features'.*"):
        ale_plot(
            model=SimpleModel(), train_set=pd.DataFrame([1]), features=list(range(3))
        )

    with pytest.raises(ValueError, match=r"'0' 'features'.*"):
        ale_plot(model=SimpleModel(), train_set=pd.DataFrame([1]), features=[])

    with pytest.raises(
        NotImplementedError, match="'features_classes' is not implemented yet."
    ):
        ale_plot(
            model=SimpleModel(),
            train_set=pd.DataFrame([1]),
            features=[0],
            features_classes=["a"],
        )

    with pytest.raises(ValueError, match=r"1 feature.*but 'bins' was not an integer."):
        ale_plot(
            model=SimpleModel(), train_set=pd.DataFrame([1]), features=[0], bins=1.0,
        )


def test_ax_quantiles():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="'twin' should be one of 'x' or 'y'."):
        _ax_quantiles(ax, list(range(2)), "z")
    plt.close(fig)

    fig, ax = plt.subplots()
    _ax_quantiles(ax, list(range(2)), "x")
    plt.close(fig)
