import numpy as np
from scipy.special import ndtr

from candel.model.utils import log_prob_integrand_window_sel


def test_log_prob_integrand_window_sel_matches_cdf_difference():
    x = np.array([21.0, 22.5, 24.0, 27.0])
    low = 22.0
    high = 25.0
    width = 0.7

    actual = np.asarray(log_prob_integrand_window_sel(
        x, 0.0, low, high, width))
    expected = np.log(
        ndtr((high - x) / width) - ndtr((low - x) / width))

    assert np.allclose(actual, expected)


def test_log_prob_integrand_window_sel_is_stable_in_positive_tail():
    actual = np.asarray(log_prob_integrand_window_sel(
        np.array([-100.0]), 0.0, 22.0, 25.0, 1.0))

    assert np.all(np.isfinite(actual))


def test_log_prob_integrand_window_sel_is_stable_for_tiny_window():
    actual = np.asarray(log_prob_integrand_window_sel(
        np.array([22.0]), 0.0, 22.0, 22.0 + 1e-8, 0.1))

    assert np.all(np.isfinite(actual))
