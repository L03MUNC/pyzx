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


def dol_to_mpl(dol, title, xlabel, ylabel, figsize=[6.4, 4.8], filepath=None):
    """
    """

    pass

def lod_to_mpl(lod, title, xlabel, ylabel, figsize=[6.4, 4.8], filepath=None):
    """
    """

    pass
