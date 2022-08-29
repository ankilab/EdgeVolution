import itertools


def grouper(n, iterable, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)