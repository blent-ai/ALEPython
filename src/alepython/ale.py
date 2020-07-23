# -*- coding: utf-8 -*-
"""ALE plotting for continuous or categorical features."""
from collections.abc import Iterable
from functools import reduce
from itertools import product
from operator import add

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from loguru import logger
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree

logger.disable("alepython")


__all__ = ("ale_plot",)


def _parse_features(features):
    """Standardise representation of column labels.

    Args:
        features : object
            One or more column labels.

    Returns:
        features : array-like
            An array of input features.

    Examples
    --------
    >>> _parse_features(1)
    array([1])
    >>> _parse_features(('a', 'b'))
    array(['a', 'b'], dtype='<U1')

    """
    if isinstance(features, Iterable) and not isinstance(features, str):
        # If `features` is a non-string iterable.
        return np.asarray(features)
    else:
        # If `features` is not an iterable, or it is a string, then assume it
        # represents one column label.
        return np.asarray([features])


def _check_two_ints(values):
    """Retrieve two integers.

    Parameters
    ----------
    values : [2-iterable of] int
        Values to process.

    Returns
    -------
    values : 2-tuple of int
        The processed integers.

    Raises
    ------
    ValueError
        If more than 2 values are given.
    ValueError
        If the values are not integers.

    Examples
    --------
    >>> _check_two_ints(1)
    (1, 1)
    >>> _check_two_ints((1, 2))
    (1, 2)
    >>> _check_two_ints((1,))
    (1, 1)

    """
    if isinstance(values, (int, np.integer)):
        values = (values, values)
    elif len(values) == 1:
        values = (values[0], values[0])
    elif len(values) != 2:
        raise ValueError(
            "'{}' values were given. Expected at most 2.".format(len(values))
        )

    if not all(isinstance(n_bin, (int, np.integer)) for n_bin in values):
        raise ValueError(
            "All values must be an integer. Got types '{}' instead.".format(
                {type(n_bin) for n_bin in values}
            )
        )
    return values


def _get_centres(x):
    """Return bin centres from bin edges.

    Parameters
    ----------
    x : array-like
        The first axis of `x` will be averaged.

    Returns
    -------
    centres : array-like
        The centres of `x`, the shape of which is (N - 1, ...) for
        `x` with shape (N, ...).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 1, 2, 3])
    >>> _get_centres(x)
    array([0.5, 1.5, 2.5])

    """
    return (x[1:] + x[:-1]) / 2


def _ax_title(ax, title, subtitle=""):
    """Add title to axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to add title to.
    title : str
        Axis title.
    subtitle : str, optional
        Sub-title for figure. Will appear one line below `title`.

    """
    ax.set_title("\n".join((title, subtitle)))


def _ax_labels(ax, xlabel=None, ylabel=None):
    """Add labels to axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to add labels to.
    xlabel : str, optional
        X axis label.
    ylabel : str, optional
        Y axis label.

    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def _ax_quantiles(ax, quantiles, twin="x"):
    """Plot quantiles of a feature onto axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to modify.
    quantiles : array-like
        Quantiles to plot.
    twin : {'x', 'y'}, optional
        Select the axis for which to plot quantiles.

    Raises
    ------
    ValueError
        If `twin` is not one of 'x' or 'y'.

    """
    if twin not in ("x", "y"):
        raise ValueError("'twin' should be one of 'x' or 'y'.")

    logger.debug("Quantiles: {}.", quantiles)

    # Duplicate the 'opposite' axis so we can define a distinct set of ticks for the
    # desired axis (`twin`).
    ax_mod = ax.twiny() if twin == "x" else ax.twinx()

    # Set the new axis' ticks for the desired axis.
    getattr(ax_mod, "set_{twin}ticks".format(twin=twin))(quantiles)
    # Set the corresponding tick labels.

    # Calculate tick label percentage values for each quantile (bin edge).
    percentages = (
        100 * np.arange(len(quantiles), dtype=np.float64) / (len(quantiles) - 1)
    )

    # If there is a fractional part, add a decimal place to show (part of) it.
    fractional = (~np.isclose(percentages % 1, 0)).astype("int8")

    getattr(ax_mod, "set_{twin}ticklabels".format(twin=twin))(
        [
            "{0:0.{1}f}%".format(percent, format_fraction)
            for percent, format_fraction in zip(percentages, fractional)
        ],
        color="#545454",
        fontsize=7,
    )
    getattr(ax_mod, "set_{twin}lim".format(twin=twin))(
        getattr(ax, "get_{twin}lim".format(twin=twin))()
    )


def _first_order_quant_plot(ax, quantiles, ale, **kwargs):
    """First order ALE plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot onto.
    quantiles : array-like
        ALE quantiles.
    ale : array-like
        ALE to plot.
    **kwargs : plot properties, optional
        Additional keyword parameters are passed to `ax.plot`.

    """
    ax.plot(_get_centres(quantiles), ale, **kwargs)


def _second_order_quant_plot(
    fig, ax, quantiles_list, ale, mark_empty=True, n_interp=50, **kwargs
):
    """Second order ALE plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot onto.
    quantiles_list : array-like
        ALE quantiles for the first (`quantiles_list[0]`) and second
        (`quantiles_list[1]`) features.
    ale : masked array
        ALE to plot. Where `ale.mask` is True, this denotes bins where no samples were
        available. See `mark_empty`.
    mark_empty : bool, optional
        If True, plot rectangles over bins that did not contain any samples.
    n_interp : [2-iterable of] int, optional
        The number of interpolated samples generated from `ale` prior to contour
        plotting. Two integers may be given to specify different interpolation steps
        for the two features.
    **kwargs : contourf properties, optional
        Additional keyword parameters are passed to `ax.contourf`.

    Raises
    ------
    ValueError
        If `n_interp` values were not integers.
    ValueError
        If more than 2 values were given for `n_interp`.

    """
    centres_list = [_get_centres(quantiles) for quantiles in quantiles_list]
    n_x, n_y = _check_two_ints(n_interp)
    x = np.linspace(centres_list[0][0], centres_list[0][-1], n_x)
    y = np.linspace(centres_list[1][0], centres_list[1][-1], n_y)

    X, Y = np.meshgrid(x, y, indexing="xy")
    ale_interp = scipy.interpolate.interp2d(centres_list[0], centres_list[1], ale.T)
    CF = ax.contourf(X, Y, ale_interp(x, y), cmap="bwr", levels=30, alpha=0.7, **kwargs)

    if mark_empty and np.any(ale.mask):
        # Do not autoscale, so that boxes at the edges (contourf only plots the bin
        # centres, not their edges) don't enlarge the plot.
        plt.autoscale(False)
        # Add rectangles to indicate cells without samples.
        for i, j in zip(*np.where(ale.mask)):
            ax.add_patch(
                Rectangle(
                    [quantiles_list[0][i], quantiles_list[1][j]],
                    quantiles_list[0][i + 1] - quantiles_list[0][i],
                    quantiles_list[1][j + 1] - quantiles_list[1][j],
                    linewidth=1,
                    edgecolor="k",
                    facecolor="none",
                    alpha=0.4,
                )
            )
    fig.colorbar(CF)


def _get_quantiles(train_set, feature, bins):
    """Get quantiles from a feature in a dataset.

    Parameters
    ----------
    train_set : pandas.core.frame.DataFrame
        Dataset containing feature `feature`.
    feature : column label
        Feature for which to calculate quantiles.
    bins : int
        The number of quantiles is calculated as `bins + 1`.

    Returns
    -------
    quantiles : array-like
        Quantiles.
    bins : int
        Number of bins, `len(quantiles) - 1`. This may be lower than the original
        `bins` if identical quantiles were present.

    Raises
    ------
    ValueError
        If `bins` is not an integer.

    Notes
    -----
    When using this definition of quantiles in combination with a half open interval
    (lower quantile, upper quantile], care has to taken that the smallest observation
    is included in the first bin. This is handled transparently by `np.digitize`.

    """
    if not isinstance(bins, (int, np.integer)):
        raise ValueError(
            "Expected integer 'bins', but got type '{}'.".format(type(bins))
        )
    quantiles = np.unique(
        np.quantile(
            train_set[feature], np.linspace(0, 1, bins + 1), interpolation="lower"
        )
    )
    bins = len(quantiles) - 1
    return quantiles, bins


def _first_order_ale_quant(predictor, train_set, feature, bins):
    """Estimate the first-order ALE function for single continuous feature data.

    Parameters
    ----------
    predictor : callable
        Prediction function.
    train_set : pandas.core.frame.DataFrame
        Training set on which the model was trained.
    feature : column label
        Feature name. A single column label.
    bins : int
        This defines the number of bins to compute. The effective number of bins may
        be less than this as only unique quantile values of train_set[feature] are
        used.

    Returns
    -------
    ale : array-like
        The first order ALE.
    quantiles : array-like
        The quantiles used.

    """
    quantiles, _ = _get_quantiles(train_set, feature, bins)
    logger.debug("Quantiles: {}.", quantiles)

    # Define the bins the feature samples fall into. Shift and clip to ensure we are
    # getting the index of the left bin edge and the smallest sample retains its index
    # of 0.
    indices = np.clip(
        np.digitize(train_set[feature], quantiles, right=True) - 1, 0, None
    )

    # Assign the feature quantile values to two copied training datasets, one for each
    # bin edge. Then compute the difference between the corresponding predictions
    predictions = []
    for offset in range(2):
        mod_train_set = train_set.copy()
        mod_train_set[feature] = quantiles[indices + offset]
        predictions.append(predictor(mod_train_set))
    # The individual effects.
    effects = predictions[1] - predictions[0]

    # Average these differences within each bin.
    index_groupby = pd.DataFrame({"index": indices, "effects": effects}).groupby(
        "index"
    )

    mean_effects = index_groupby.mean().to_numpy().flatten()

    ale = np.array([0, *np.cumsum(mean_effects)])

    # The uncentred mean main effects at the bin centres.
    ale = _get_centres(ale)

    # Centre the effects by subtracting the mean (the mean of the individual
    # `effects`, which is equivalently calculated using `mean_effects` and the number
    # of samples in each bin).
    ale -= np.sum(ale * index_groupby.size() / train_set.shape[0])
    return ale, quantiles


def _second_order_ale_quant(predictor, train_set, features, bins):
    """Estimate the second-order ALE function for two continuous feature data.

    Parameters
    ----------
    predictor : callable
        Prediction function.
    train_set : pandas.core.frame.DataFrame
        Training set on which the model was trained.
    features : 2-iterable of column label
        The two desired features, as two column labels.
    bins : [2-iterable of] int
        This defines the number of bins to compute. The effective number of bins may
        be less than this as only unique quantile values of train_set[feature] are
        used. If one integer is given, this is used for both features.

    Returns
    -------
    ale : (M, N) masked array
        The second order ALE. Elements are masked where no data was available.
    quantiles : 2-tuple of array-like
        The quantiles used: first the quantiles for `features[0]` with shape (M + 1,),
        then for `features[1]` with shape (N + 1,).

    Raises
    ------
    ValueError
        If `features` does not contain 2 features.
    ValueError
        If more than 2 bins are given.
    ValueError
        If bins are not integers.

    """
    features = _parse_features(features)
    if len(features) != 2:
        raise ValueError(
            "'features' contained '{n_feat}' features. Expected 2.".format(
                n_feat=len(features)
            )
        )

    quantiles_list, bins_list = tuple(
        zip(
            *(
                _get_quantiles(train_set, feature, n_bin)
                for feature, n_bin in zip(features, _check_two_ints(bins))
            )
        )
    )
    logger.debug("Quantiles: {}.", quantiles_list)

    # Define the bins the feature samples fall into. Shift and clip to ensure we are
    # getting the index of the left bin edge and the smallest sample retains its index
    # of 0.
    indices_list = [
        np.clip(np.digitize(train_set[feature], quantiles, right=True) - 1, 0, None)
        for feature, quantiles in zip(features, quantiles_list)
    ]

    # Invoke the predictor at the corners of the bins. Then compute the second order
    # difference between the predictions at the bin corners.
    predictions = {}
    for shifts in product(*(range(2),) * 2):
        mod_train_set = train_set.copy()
        for i in range(2):
            mod_train_set[features[i]] = quantiles_list[i][indices_list[i] + shifts[i]]
        predictions[shifts] = predictor(mod_train_set)
    # The individual effects.
    effects = (predictions[(1, 1)] - predictions[(1, 0)]) - (
        predictions[(0, 1)] - predictions[(0, 0)]
    )

    # Group the effects by their indices along both axes.
    index_groupby = pd.DataFrame(
        {"index_0": indices_list[0], "index_1": indices_list[1], "effects": effects}
    ).groupby(["index_0", "index_1"])

    # Compute mean effects.
    mean_effects = index_groupby.mean()
    # Get the indices of the mean values.
    group_indices = mean_effects.index
    valid_grid_indices = tuple(zip(*group_indices))
    # Extract only the data.
    mean_effects = mean_effects.to_numpy().flatten()

    # Get the number of samples in each bin.
    n_samples = index_groupby.size().to_numpy()

    # Create a 2D array of the number of samples in each bin.
    samples_grid = np.zeros(bins_list)
    samples_grid[valid_grid_indices] = n_samples

    ale = np.ma.MaskedArray(
        np.zeros((len(quantiles_list[0]), len(quantiles_list[1]))),
        mask=np.ones((len(quantiles_list[0]), len(quantiles_list[1]))),
    )
    # Mark the first row/column as valid, since these are meant to contain 0s.
    ale.mask[0, :] = False
    ale.mask[:, 0] = False

    # Place the mean effects into the final array.
    # Since `ale` contains `len(quantiles)` rows/columns the first of which are
    # guaranteed to be valid (and filled with 0s), ignore the first row and column.
    ale[1:, 1:][valid_grid_indices] = mean_effects

    # Record where elements were missing.
    missing_bin_mask = ale.mask.copy()[1:, 1:]

    if np.any(missing_bin_mask):
        # Replace missing entries with their nearest neighbours.

        # Calculate the dense location matrices (for both features) of all bin centres.
        centres_list = np.meshgrid(
            *(_get_centres(quantiles) for quantiles in quantiles_list), indexing="ij"
        )

        # Select only those bin centres which are valid (had observation).
        valid_indices_list = np.where(~missing_bin_mask)
        tree = cKDTree(
            np.hstack(
                tuple(
                    centres[valid_indices_list][:, np.newaxis]
                    for centres in centres_list
                )
            )
        )

        row_indices = np.hstack(
            [inds.reshape(-1, 1) for inds in np.where(missing_bin_mask)]
        )
        # Select both columns for each of the rows above.
        column_indices = np.hstack(
            (
                np.zeros((row_indices.shape[0], 1), dtype=np.int8),
                np.ones((row_indices.shape[0], 1), dtype=np.int8),
            )
        )

        # Determine the indices of the points which are nearest to the empty bins.
        nearest_points = tree.query(tree.data[row_indices, column_indices])[1]

        nearest_indices = tuple(
            valid_indices[nearest_points] for valid_indices in valid_indices_list
        )

        # Replace the invalid bin values with the nearest valid ones.
        ale[1:, 1:][missing_bin_mask] = ale[1:, 1:][nearest_indices]

    # Compute the cumulative sums.
    ale = np.cumsum(np.cumsum(ale, axis=0), axis=1)

    # Subtract first order effects along both axes separately.
    for i in range(2):
        # Depending on `i`, reverse the arguments to operate on the opposite axis.
        flip = slice(None, None, 1 - 2 * i)

        # Undo the cumulative sum along the axis.
        first_order = ale[(slice(1, None), ...)[flip]] - ale[(slice(-1), ...)[flip]]
        # Average the diffs across the other axis.
        first_order = (
            first_order[(..., slice(1, None))[flip]]
            + first_order[(..., slice(-1))[flip]]
        ) / 2
        # Weight by the number of samples in each bin.
        first_order *= samples_grid
        # Take the sum along the axis.
        first_order = np.sum(first_order, axis=1 - i)
        # Normalise by the number of samples in the bins along the axis.
        first_order /= np.sum(samples_grid, axis=1 - i)
        # The final result is the cumulative sum (with an additional 0).
        first_order = np.array([0, *np.cumsum(first_order)]).reshape((-1, 1)[flip])

        # Subtract the first order effect.
        ale -= first_order

    # Compute the ALE at the bin centres.
    ale = (
        reduce(
            add,
            (
                ale[i : ale.shape[0] - 1 + i, j : ale.shape[1] - 1 + j]
                for i, j in list(product(*(range(2),) * 2))
            ),
        )
        / 4
    )

    # Centre the ALE by subtracting its expectation value.
    ale -= np.sum(samples_grid * ale) / train_set.shape[0]

    # Mark the originally missing points as such to enable later interpretation.
    ale.mask = missing_bin_mask
    return ale, quantiles_list


def _first_order_ale_cat(
    predictor, train_set, feature, features_classes, feature_encoder=None
):
    """Compute the first-order ALE function on single categorical feature data.

    Parameters
    ----------
    predictor : callable
        Prediction function.
    train_set : pandas.core.frame.DataFrame
        Training set on which model was trained.
    feature : str
        Feature name.
    features_classes : iterable or str
        Feature's classes.
    feature_encoder : callable or iterable
        Encoder that was used to encode categorical feature. If features_classes is
        not None, this parameter is skipped.

    """
    num_cat = len(features_classes)
    ale = np.zeros(num_cat)  # Final ALE function.

    for i in range(num_cat):
        subset = train_set[train_set[feature] == features_classes[i]]

        # Without any observation, local effect on split area is null.
        if len(subset) != 0:
            z_low = subset.copy()
            z_up = subset.copy()
            # The main ALE idea that compute prediction difference between same data
            # except feature's one.
            z_low[feature] = quantiles[i - 1]
            z_up[feature] = quantiles[i]
            ale[i] += (predictor(z_up) - predictor(z_low)).sum() / subset.shape[0]

    # The accumulated effect.
    ale = ale.cumsum()
    # Now we have to center ALE function in order to obtain null expectation for ALE
    # function.
    ale -= ale.mean()
    return ale


def ale_plot(
    model,
    train_set,
    features,
    bins=10,
    monte_carlo=False,
    predictor=None,
    features_classes=None,
    monte_carlo_rep=50,
    monte_carlo_ratio=0.1,
    rugplot_lim=1000,
):
    """Plots ALE function of specified features based on training set.

    Parameters
    ----------
    model : object
        An object that implements a 'predict' method. If None, a `predictor` function
        must be supplied which will be used instead of `model.predict`.
    train_set : pandas.core.frame.DataFrame
        Training set on which model was trained.
    features : [2-iterable of] column label
        One or two features for which to plot the ALE plot.
    bins : [2-iterable of] int, optional
        Number of bins used to split feature's space. 2 integers can only be given
        when 2 features are supplied in order to compute a different number of
        quantiles for each feature.
    monte_carlo : boolean, optional
        Compute and plot Monte-Carlo samples.
    predictor : callable
        Custom prediction function. See `model`.
    features_classes : iterable of str, optional
        If features is first-order and a categorical variable, plot ALE according to
        discrete aspect of data.
    monte_carlo_rep : int
        Number of Monte-Carlo replicas.
    monte_carlo_ratio : float
        Proportion of randomly selected samples from dataset for each Monte-Carlo
        replica.
    rugplot_lim : int, optional
        If `train_set` has more rows than `rugplot_lim`, no rug plot will be plotted.
        Set to None to always plot rug plots. Set to 0 to always plot rug plots.

    Raises
    ------
    ValueError
        If both `model` and `predictor` are None.
    ValueError
        If `len(features)` not in {1, 2}.
    ValueError
        If multiple bins were given for 1 feature.
    NotImplementedError
        If `features_classes` is not None.

    """
    if model is None and predictor is None:
        raise ValueError("If 'model' is None, 'predictor' must be supplied.")

    if features_classes is not None:
        raise NotImplementedError("'features_classes' is not implemented yet.")

    fig, ax = plt.subplots()

    features = _parse_features(features)

    if len(features) == 1:
        if not isinstance(bins, (int, np.integer)):
            raise ValueError("1 feature was given, but 'bins' was not an integer.")

        if features_classes is None:
            # Continuous data.

            if monte_carlo:
                mc_replicates = np.asarray(
                    [
                        [
                            np.random.choice(range(train_set.shape[0]))
                            for _ in range(int(monte_carlo_ratio * train_set.shape[0]))
                        ]
                        for _ in range(monte_carlo_rep)
                    ]
                )
                for k, rep in enumerate(mc_replicates):
                    train_set_rep = train_set.iloc[rep, :]
                    # Make this recursive?
                    if features_classes is None:
                        # The same quantiles cannot be reused here as this could cause
                        # some bins to be empty or contain disproportionate numbers of
                        # samples.
                        mc_ale, mc_quantiles = _first_order_ale_quant(
                            model.predict if predictor is None else predictor,
                            train_set_rep,
                            features[0],
                            bins,
                        )
                        _first_order_quant_plot(
                            ax, mc_quantiles, mc_ale, color="#1f77b4", alpha=0.06
                        )

            ale, quantiles = _first_order_ale_quant(
                model.predict if predictor is None else predictor,
                train_set,
                features[0],
                bins,
            )
            _ax_labels(ax, "Feature '{}'".format(features[0]), "")
            _ax_title(
                ax,
                "First-order ALE of feature '{0}'".format(features[0]),
                "Bins : {0} - Monte-Carlo : {1}".format(
                    len(quantiles) - 1,
                    mc_replicates.shape[0] if monte_carlo else "False",
                ),
            )
            ax.grid(True, linestyle="-", alpha=0.4)
            if rugplot_lim is None or train_set.shape[0] <= rugplot_lim:
                sns.rugplot(train_set[features[0]], ax=ax, alpha=0.2)
            _first_order_quant_plot(ax, quantiles, ale, color="black")
            _ax_quantiles(ax, quantiles)

    elif len(features) == 2:
        if features_classes is None:
            # Continuous data.
            ale, quantiles_list = _second_order_ale_quant(
                model.predict if predictor is None else predictor,
                train_set,
                features,
                bins,
            )
            _second_order_quant_plot(fig, ax, quantiles_list, ale)
            _ax_labels(
                ax,
                "Feature '{}'".format(features[0]),
                "Feature '{}'".format(features[1]),
            )
            for twin, quantiles in zip(("x", "y"), quantiles_list):
                _ax_quantiles(ax, quantiles, twin=twin)
            _ax_title(
                ax,
                "Second-order ALE of features '{0}' and '{1}'".format(
                    features[0], features[1]
                ),
                "Bins : {0}x{1}".format(*[len(quant) - 1 for quant in quantiles_list]),
            )
    else:
        raise ValueError(
            "'{n_feat}' 'features' were given, but only up to 2 are supported.".format(
                n_feat=len(features)
            )
        )
    plt.show()
    return ax
