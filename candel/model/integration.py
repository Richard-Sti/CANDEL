"""
Routines copied from:
    `https://github.com/adrn/jax-ext/blob/main/jax_ext/integrate/simpson.py`

Only minor changes were made to the original code.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def tupleset(t, i, value):
    _l = list(t)
    _l[i] = value
    return tuple(_l)


@partial(jax.jit, static_argnums=(1, 3))
def _basic_ln_simpson(ln_y, stop, x, axis):
    """
    Note: Interface comes from scipy.integrate.simpson implementation
    """
    nd = len(ln_y.shape)  # number of dimensions
    step = 2

    slice_all = (slice(None),) * nd
    slice0 = tupleset(slice_all, axis, slice(0, stop, step))
    slice1 = tupleset(slice_all, axis, slice(1, stop + 1, step))
    slice2 = tupleset(slice_all, axis, slice(2, stop + 2, step))

    # Account for possibly different spacings.
    h = jnp.diff(x, axis=axis)
    h0 = h[slice0]
    h1 = h[slice1]
    hsum = h0 + h1
    hprod = h0 * h1
    h0divh1 = jnp.where(h1 != 0, h0 / h1, 0.0)

    term = logsumexp(
        jnp.array([ln_y[slice0], ln_y[slice1], ln_y[slice2]]),
        b=jnp.array(
            [
                (2.0 - jnp.where(h0divh1 != 0, 1 / h0divh1, 0.0)),
                (hsum * jnp.where(hprod != 0, hsum / hprod, 0.0)),
                (2.0 - h0divh1),
            ]
        ),
        axis=0,
    )
    tmp = jnp.log(hsum / 6.0) + term
    return logsumexp(tmp, axis=axis)


def simpson_log_weights(x):
    """Log of composite Simpson's 1/3 weights for a 1D grid.

    For N points (N odd, >= 3), the composite Simpson rule is
        ∫ f dx ≈ Σ_i w_i f(x_i)
    This returns log(w_i). Requires all weights positive (true when
    adjacent spacing ratios stay below 2).
    """
    x = jnp.asarray(x)
    h = jnp.diff(x)
    h0, h1 = h[0::2], h[1::2]
    hsum = h0 + h1

    c_left = hsum / 6 * (2 - h1 / h0)
    c_mid = hsum**3 / (6 * h0 * h1)
    c_right = hsum / 6 * (2 - h0 / h1)

    N = x.shape[0]
    n_panels = len(h0)
    idx = jnp.arange(n_panels) * 2
    w = jnp.zeros(N)
    w = w.at[idx].add(c_left)
    w = w.at[idx + 1].add(c_mid)
    w = w.at[idx + 2].add(c_right)

    return jnp.log(w)


@partial(jax.jit, static_argnums=2)
def ln_simpson_precomputed(ln_y, log_w, axis=-1):
    """Log-space Simpson integration using pre-computed log weights.

    Equivalent to ``ln_simpson`` but avoids recomputing weights each call.
    Use ``simpson_log_weights(x)`` once to obtain ``log_w``.

    Parameters
    ----------
    ln_y : array_like
        Log function values. The integration axis must match `log_w`.
    log_w : array_like
        Pre-computed log Simpson weights from ``simpson_log_weights``.
    axis : int
        Axis along which to integrate.
    """
    return logsumexp(ln_y + log_w, axis=axis)


@partial(jax.jit, static_argnums=2)
def ln_simpson(ln_y, x, axis=-1):
    """
    Integrate y(x) using log values of the function `ln_y` evaluated at the
    locations `x`, and return the log of the integral

    Note: `x` values must be increasing and `x` and `ln_y` must have the
    same length.

    Parameters
    ----------
    ln_y : array_like
        Array of log function values to be integrated.
    x : array_like
        The points at which `ln_y` is evaluated.

    Returns
    -------
    float
        The estimated log integral computed with the composite Simpson's rule.
    """
    ln_y = jnp.asarray(ln_y)
    x = jnp.asarray(x)
    N = ln_y.shape[axis]

    if N % 2 == 0:
        raise ValueError("Even number of integration points is not supported.")

    # Clip relative to the maximum value and shift all log values
    max_ln_y = jnp.max(ln_y, axis=axis, keepdims=True)
    ln_y = ln_y - max_ln_y
    result = _basic_ln_simpson(ln_y, N - 2, x, axis=axis)

    # Summing over axis drops that dimension, so we need to squeeze shift
    return result + jnp.squeeze(max_ln_y, axis=axis)
