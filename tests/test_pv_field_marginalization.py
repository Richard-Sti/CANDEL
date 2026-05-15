import numpy as np
from jax.scipy.special import logsumexp

from candel.model.base_pv import field_product_logmeanexp


def test_field_product_logmeanexp_products_objects_before_field_average():
    ll = np.array([
        [-1.0, -8.0, -2.0],
        [-3.0, -1.5, -4.0],
        [-0.5, -6.0, -5.5],
    ])

    expected = logsumexp(ll.sum(axis=1), axis=0) - np.log(ll.shape[0])
    old_order = (logsumexp(ll, axis=0) - np.log(ll.shape[0])).sum()

    actual = field_product_logmeanexp(ll, ll.shape[0])

    assert np.allclose(actual, expected)
    assert not np.allclose(actual, old_order)
