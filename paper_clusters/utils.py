"""Paper-specific utility functions."""
import math
import os


def stem_from_fname(fname: str) -> str:
    """Extract stem (filename without extension) from a path."""
    return os.path.basename(fname).split(".hdf5")[0]


def pm_two_dec(val, err=None):
    """Format a value with optional error to 2 decimal places."""
    if val is None:
        return r"\textemdash"
    if err is None:
        return f"{val:.2f}"
    return f"{val:.2f} $\\pm$ {err:.2f}"


def quadrature(a, b):
    """Add two values in quadrature, handling None values."""
    if a is None and b is None:
        return None
    a = 0.0 if a is None else a
    b = 0.0 if b is None else b
    return math.sqrt(a * a + b * b)


def safe_get(read_gof_func, fname, key):
    """Safely get a GOF statistic, returning None on failure."""
    try:
        return read_gof_func(fname, key)
    except Exception:
        return None
