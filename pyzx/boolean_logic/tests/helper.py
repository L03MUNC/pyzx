import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def printv(*args, verbosity, level, **kwargs):
    """
    Print to stderr if verbosity is high enough.

    Parameters
    ----------
    *args
        Arguments to print.
    verbosity : int
        Current verbosity level.
    level : int
        Minimum verbosity level for printing.
    **kwargs
        Additional keyword arguments for the print function.
    """

    if verbosity >= level:
        kwargs['file'] = kwargs.get('file') or sys.stderr
        print(*args, **kwargs)


def dol_from_lod(lod, key_key, list_key, list_index_range=None):
    """Group a list of dicts into counts stored as a dict of lists.

    Parameters
    ----------
    lod : list[dict]
        List of dictionaries.
    key_key : hashable
        Dictionary key used for grouping (values become the output dict keys).
    list_key : hashable
        Dictionary key whose (integer) value selects a list index.
    list_index_range : tuple[int, int], optional
        Inclusive ``(min, max)`` range of possible ``list_key`` values.
        If omitted, the range is inferred from ``lod``.

    Returns
    -------
    dict
        Mapping ``group_value -> counts`` where ``counts`` is a list of length
        ``max-min+1`` and counts are stored at index ``value - min``.

    Raises
    ------
    KeyError
        If any dictionary in ``lod`` is missing either ``key_key`` or ``list_key``.
    """

    if list_index_range:
        list_index_min, list_index_max = list_index_range
    else:
        list_index_min = min(dictionary[list_key] for dictionary in lod)
        list_index_max = max(dictionary[list_key] for dictionary in lod)
    result = dict()
    for dictionary in lod:
        k = dictionary[key_key]
        if k not in result:
            result[k] = [0] * (list_index_max - list_index_min + 1)
        result[k][dictionary[list_key] - list_index_min] += 1
        
    return result

def lod_from_lod(lod, key_key, list_key, list_index_range=None):
    """Group a list of dicts into counts stored as a list of dicts.

    Parameters
    ----------
    lod : list[dict]
        List of dictionaries.
    key_key : hashable
        Dictionary key used for grouping (values become the output dict keys).
    list_key : hashable
        Dictionary key whose (integer) value selects the output list index.
    list_index_range : tuple[int, int], optional
        Inclusive ``(min, max)`` range of possible ``list_key`` values.
        If omitted, the range is inferred from ``lod``.

    Returns
    -------
    list[dict]
        List of length ``max-min+1`` where element ``i`` is a dictionary
        mapping ``group_value -> count`` for ``list_key == i + min``.

    Raises
    ------
    KeyError
        If any dictionary in ``lod`` is missing either ``key_key`` or ``list_key``.
    """

    if list_index_range:
        list_index_min, list_index_max = list_index_range
    else:
        list_index_min = min(dictionary[list_key] for dictionary in lod)
        list_index_max = max(dictionary[list_key] for dictionary in lod)
    result = [dict() for _ in range(list_index_max - list_index_min + 1)]
    for dictionary in lod:
        k = dictionary[key_key]
        d = result[dictionary[list_key] - list_index_min]
        if k not in d:
            d[k] = 0
        d[k] += 1

    return result

def dodol_from_lod(lod, key_keys, list_key, list_index_range=None):
    """Compute multiple :func:`dol_from_lod` groupings for different keys.

    Parameters
    ----------
    lod : list[dict]
        List of dictionaries.
    key_keys : sequence
        Sequence of dictionary keys to group by.
    list_key : hashable
        Dictionary key whose (integer) value selects a list index.
    list_index_range : tuple[int, int], optional
        Inclusive ``(min, max)`` range of possible ``list_key`` values.

    Returns
    -------
    dict
        Mapping ``key_key -> dol`` where each ``dol`` is the return value of
        :func:`dol_from_lod(lod, key_key, list_key, list_index_range)`.

    Raises
    ------
    KeyError
        If any dictionary in ``lod`` is missing any of the keys in ``key_keys``
        or ``list_key``.
    """

    return {
        key_key: dol_from_lod(lod, key_key, list_key, list_index_range)
        for key_key in key_keys
    }

def dolod_from_lod(lod, key_keys, list_key, list_index_range=None):
    """Compute multiple :func:`lod_from_lod` groupings for different keys.

    Parameters
    ----------
    lod : list[dict]
        List of dictionaries.
    key_keys : sequence
        Sequence of dictionary keys to group by.
    list_key : hashable
        Dictionary key whose (integer) value selects a list index.
    list_index_range : tuple[int, int], optional
        Inclusive ``(min, max)`` range of possible ``list_key`` values.

    Returns
    -------
    dict
        Mapping ``key_key -> lod`` where each ``lod`` is the return value of
        :func:`lod_from_lod(lod, key_key, list_key, list_index_range)`.

    Raises
    ------
    KeyError
        If any dictionary in ``lod`` is missing any of the keys in ``key_keys``
        or ``list_key``.
    """

    return {
        key_key: lod_from_lod(lod, key_key, list_key, list_index_range)
        for key_key in key_keys
    }


def plot_metric_counts(ax, lod, metric, result_names):
    """Plot metric counts per result category.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot into.
    lod : list[dict]
        List of dictionaries, one per result category, mapping
        ``metric_value -> count``. Typically produced by
        :func:`lod_from_lod` and ordered to match ``result_names``.
    metric : str
        Name of the metric used for axis labels and titles.
    result_names : sequence of str
        Labels for the different result categories.

    Returns
    -------
    None
    """

    keys = set().union(*(result.keys() for result in lod))
    metric_min, metric_max = min(keys), max(keys)
    for result, vals in zip(result_names, lod):
        y = [vals.get(key, 0) for key in range(metric_min, metric_max + 1)]
        ax.plot(y, label=result)
    ax.legend()
    ax.set_title(f'Number of occurrences versus {metric}')
    ax.set_xlabel(metric)
    ax.set_ylabel('Number of occurrences')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

def plot_multiple_metric_counts(dolod, result_names, figsize=[6.4, 4.8], filepath=None):
    """Plot metric counts in separate figures.

    Parameters
    ----------
    dolod : dict
        Mapping ``metric -> lod`` where each ``lod`` is a list of dictionaries
        as expected by :func:`plot_metric_counts`.
    result_names : sequence of str
        Labels for the different result categories.
    figsize : list[float, float], optional
        Figure size passed to :func:`matplotlib.pyplot.subplots`.
    filepath : str or pathlib.Path, optional
        If provided, each figure is saved under ``filepath``, which should
        contain a ``{metric}`` placeholder that is replaced with the metric
        name for each plot.

    Returns
    -------
    list[tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]]
        One ``(fig, ax)`` pair per metric in ``dolod``.
    """

    fig_ax_pairs = [plt.subplots(figsize=figsize) for _ in dolod]
    for (metric, lod), (fig, ax) in zip(dolod.items(), fig_ax_pairs):
        plot_metric_counts(ax, lod, metric, result_names)
        if filepath:
            fig.savefig(str(filepath).format(metric=metric))

    return fig_ax_pairs

def plot_composed_metric_counts(dolod, result_names, component_figsize=[6.4, 4.8], filepath=None):
    """Plot multiple metric-count charts in a single composed figure.

    Parameters
    ----------
    dolod : dict
        Mapping ``metric -> lod`` where each ``lod`` is a list of dictionaries
        as expected by :func:`plot_metric_counts`.
    result_names : sequence of str
        Labels for the different result categories.
    component_figsize : list[float, float], optional
        Size of each subplot component in inches. The total figure size is
        derived from the computed grid dimensions.
    filepath : str or pathlib.Path, optional
        If provided, save the composed figure via ``fig.savefig``.

    Returns
    -------
    tuple
        ``(fig, axes)`` where ``fig`` is the created figure and ``axes`` is a
        list of axes (one per metric).
    """

    n = len(dolod)
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    figsize = [component_figsize[0] * cols, component_figsize[1] * rows]
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_flat = list(np.array(axes).flatten())
    for i in range(n, rows*cols):
        fig.delaxes(axes_flat[i])
        axes_flat.pop()
    fig.tight_layout(pad=3.0)

    for (metric, lod), ax in zip(dolod.items(), axes_flat):
        plot_metric_counts(ax, lod, metric, result_names)

    if filepath:
        fig.savefig(filepath)
    return fig, axes_flat

def plot_runtimes_ecdf(ax, runtimes, result_names, title='ECDF of runtimes'):
    """Plot empirical CDFs (ECDFs) of runtimes per result category.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot into.
    runtimes : sequence of sequence of float
        Runtime samples in milliseconds. Each inner sequence corresponds to a
        result category and is paired with ``result_names``.
    result_names : sequence of str
        Labels for the different result categories.
    title : str, optional
        Title for the plot.

    Returns
    -------
    None
    """

    for result, vals in zip(result_names, runtimes):
        if vals:
            ax.ecdf(vals, label=result)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Runtime (ms)')
    ax.set_ylabel('Cumulative probability')

def plot_runtimes_ecdf_fig(runtimes, result_names, figsize=[6.4, 4.8], title='ECDF of runtimes', filepath=None):
    """Create a new figure and plot ECDFs of runtimes.

    Parameters
    ----------
    runtimes : sequence of sequence of float
        Runtime samples in milliseconds. Each inner sequence corresponds to a
        result category and is paired with ``result_names``.
    result_names : sequence of str
        Labels for the different result categories.
    figsize : list[float, float], optional
        Figure size passed to :func:`matplotlib.pyplot.subplots`.
    title : str, optional
        Title for the plot.
    filepath : str or pathlib.Path, optional
        If provided, save the figure via ``fig.savefig``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created ``(fig, ax)``.
    """

    fig, ax = plt.subplots(figsize=figsize)
    plot_runtimes_ecdf(ax, runtimes, result_names, title=title)
    if filepath:
        fig.savefig(filepath)
    return fig, ax
