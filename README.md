
[![Build Status](https://travis-ci.org/MaximeJumelle/ALEPython.svg?branch=dev)](https://travis-ci.org/MaximeJumelle/ALEPython)

Python Accumulated Local Effects package.

# Why ALE ?

Explaining models predictions is very common when you have to deploy on a large scale a Machine Learning algorithm. As there are many methods that helps us to understand our model, one which was used for many years was Partial Dependency Plots (PDP). However, they suffer from a serious assumption that is made : **features have to be uncorrelated**. In real world scenarios, features are often correlated, whether because some are directly computed from others, or because observed phenomena produces correlated distributions.

Accumulated Local Effects (or ALE) reasonably palliates this issue as, instead of considering each marginal distribution of features, actual conditional marginal distributions are used, which is more reliable when you encounter correlated variables, even strongly ones.

This package aims to provide useful and quick access to ALE plots, so that you can easily explain your model throught predictions.

# Install

ALEPython is supported on Python 3.4, 3.5 and 3.6. You can either install package via `pip`.

```
pip install alepython
```

Or directly from sources 

```
git clone https://github.com/MaximeJumelle/ALEPython.git
cd ALEPython
pip install -r requirements.txt
python setup.py install
```

# Usage

```python
# Plots ALE of feature 'cont' with Monte-Carlo replicas (default : 50)
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

If you are interested in the project, I would be happy to collaborate with you since there are still quite a lot of improvements needed.