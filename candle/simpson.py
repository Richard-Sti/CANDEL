"""
Routines copied from:
    `https://github.com/adrn/jax-ext/blob/main/jax_ext/integrate/simpson.py`

Only minor changes were made to the original code.
"""
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jax.typing as JT
from jax.scipy.special import logsumexp


def tupleset(t: tuple, i: int, value: Any) -> tuple:
    _l = list(t)
    _l[i] = value
    return tuple(_l)


@partial(jax.jit, static_argnums=(1, 3))
def _basic_ln_simpson(ln_y: jax.Array, stop: int, x: jax.Array, axis: int) -> jax.Array:  # noqa
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
    #    Simpson's rule changes a bit.
    # print("x ", x)
    # print("x.shape ", x.shape)
    # print("slice0 ", slice0)
    # print("slice1 ", slice1)
    # print("slice2 ", slice2)

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


@partial(jax.jit, static_argnums=2)
def ln_simpson(
    ln_y: JT.ArrayLike, x: JT.ArrayLike, axis: int = -1
) -> jax.Array | float:
    """
    Integrate y(x) using log values of the function `ln_y` evaluated at the
    locations `x`, and return the log of the integral

    If there are an even number of samples, N, then there are an odd number of
    intervals (N-1), but Simpson's rule requires an even number of intervals.
    In this case, the implementation uses Simpson's rule for the first N-2
    intervals with the addition of a 3-point parabolic segment for the last
    interval using equations outlined by Cartwright. See the docstring for
    `scipy.integrate.simpson` for more information.

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
    ln_y = jnp.array(ln_y)
    x = jnp.array(x)
    nd = len(ln_y.shape)
    N = ln_y.shape[axis]

    # Clip relative to the maximum value and shift all log values
    max_ln_y = jnp.max(ln_y, axis=axis, keepdims=True)
    ln_y = jnp.clip(ln_y, max_ln_y - 30)
    # Shift all log values by subtracting the maximum
    shift = max_ln_y - 30
    ln_y -= shift

    if N % 2 == 0:  # Even number of points, odd intervals
        slice_all = (slice(None),) * nd

        if N == 2:
            # need at least 3 points in integration axis to form parabolic
            # segment
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            last_dx = x[slice1] - x[slice2]
            result = (jnp.log(0.5 * last_dx)
                      + jnp.logaddexp(ln_y[slice1], ln_y[slice2]))

        else:
            # use Simpson's rule on first intervals
            result = _basic_ln_simpson(ln_y, N - 3, x, axis=axis)

            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            slice3 = tupleset(slice_all, axis, -3)

            # grab the last two spacings from the appropriate axis
            hm2 = tupleset(slice_all, axis, slice(-2, -1, 1))
            hm1 = tupleset(slice_all, axis, slice(-1, None, 1))

            diffs = jnp.diff(x, axis=axis)
            h0 = jnp.squeeze(diffs[hm2], axis=axis)
            h1 = jnp.squeeze(diffs[hm1], axis=axis)

            # This is the correction for the last interval according to
            # Cartwright.
            # However, I used the equations given at
            # https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data  # noqa
            # A footnote on Wikipedia says:
            # Cartwright 2017, Equation 8. The equation in Cartwright is
            # calculating the first interval whereas the equations in the
            # Wikipedia article are adjusting for the last integral. If the
            # proper algebraic substitutions are made, the equation results in
            # the values shown.
            num = 2 * h1**2 + 3 * h0 * h1
            den = 6 * (h1 + h0)
            alpha = jnp.where(den != 0, num / den, 0.0)

            num = h1**2 + 3.0 * h0 * h1
            den = 6 * h0
            beta = jnp.where(den != 0, num / den, 0.0)

            num = 1 * h1**3
            den = 6 * h0 * (h0 + h1)
            eta = jnp.where(den != 0, num / den, 0.0)

            term = logsumexp(
                jnp.array([ln_y[slice1], ln_y[slice2], ln_y[slice3]]),
                b=jnp.array([alpha, beta, -eta]),
                axis=0,
            )
            result = jnp.logaddexp(result, term)

    else:
        result = _basic_ln_simpson(ln_y, N - 2, x, axis=axis)

    # Summing over axis drops that dimension, so we need to squeeze shift
    return result + jnp.squeeze(shift, axis=axis)
