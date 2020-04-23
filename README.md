[![Build Status](https://travis-ci.org/MaximeJumelle/ALEPython.svg?branch=dev)](https://travis-ci.org/MaximeJumelle/ALEPython)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Python Accumulated Local Effects package.

# Why ALE?

Explaining model predictions is very common when you have to deploy a Machine Learning algorithm on a large scale.
There are many methods that help us understand our model; one these uses Partial Dependency Plots (PDP), which have been widely used for years.

However, they suffer from a stringent assumption: **features have to be uncorrelated**.
In real world scenarios, features are often correlated, whether because some are directly computed from others, or because observed phenomena produce correlated distributions.

Accumulated Local Effects (or ALE) plots first proposed by [_Apley and Zhu_ (2016)](#1) alleviate this issue reasonably by using actual conditional marginal distributions instead of considering each marginal distribution of features.
This is more reliable when handling (even strongly) correlated variables.

This package aims to provide useful and quick access to ALE plots, so that you can easily explain your model through predictions.

For further details about model interpretability and ALE plots, see eg. [_Molnar_ (2020)](#2).

# Install

ALEPython is supported on Python >= 3.5.
You can either install package via `pip`:

```sh
pip install alepython
```
directly from source (including requirements):
```sh
pip install git+https://github.com/MaximeJumelle/ALEPython.git@dev#egg=alepython
```
or after cloning (or forking) for development purposes, including test dependencies:
```sh
git clone https://github.com/MaximeJumelle/ALEPython.git
pip install -e "ALEPython[test]"
```

# Usage

```python
from alepython import ale_plot
# Plots ALE of feature 'cont' with Monte-Carlo replicas (default : 50).
ale_plot(model, X_train, 'cont', monte_carlo=True)
```

# Highlights

- First-order ALE plots of continuous features
- Second-order ALE plots of continuous features

# Gallery

## First-order ALE plots of continuous features

<center><img src='https://github.com/MaximeJumelle/ALEPython/raw/dev/resources/fo_ale_quant.png'></center>

---

## Second-order ALE plots of continuous features

<center><img src='https://github.com/MaximeJumelle/ALEPython/raw/dev/resources/so_ale_quant.png'></center>

# Work In Progress

- First-order ALE plots of categorical features
- Enhanced visualization of first-order plots
- Second-order ALE plots of categorical features
- Documentation and API reference
- Jupyter Notebook examples
- Upload to PyPi
- Upload to conda-forge
- Use of matplotlib styles or kwargs to allow overriding plotting appearance

If you are interested in the project, I would be happy to collaborate with you since there are still quite a lot of improvements needed.

## References

<a id="1"></a>
Apley, Daniel W., and Jingyu Zhu. 2016. Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models. <https://arxiv.org/abs/1612.08468>.

<a id="2"></a>
Molnar, Christoph. 2020. Interpretable Machine Learning. <https://christophm.github.io/interpretable-ml-book/>.
