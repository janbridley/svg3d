import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as np_strategies

from svg3d import _pad_arrays
from svg3d.utils import _stable_normalize


def test_pad_arrays(random_ragged_array):
    subarray_count = len(random_ragged_array)
    subarray_max_len = max(len(arr) for arr in random_ragged_array)

    assert _pad_arrays(random_ragged_array).shape == (
        subarray_count,
        subarray_max_len,
        3,
    )


@pytest.mark.parametrize("shape", [(1,), (2,), (3,), (5,)])
def test_stable_normalize_zero_vector(shape):
    """Zero vector should return zero vector for various dimensions."""
    vec = np.zeros(shape)
    result = _stable_normalize(vec)
    np.testing.assert_array_equal(result, vec)


@given(
    np_strategies.arrays(
        dtype=np.float64,
        shape=(3,),
        elements=st.floats(allow_nan=False, allow_infinity=False),
    )
)
def test_stable_normalize_3d_vector(vec):
    """Normalized result should have unit length for non-zero 3D vectors."""
    result = _stable_normalize(vec)
    if np.any(np.abs(vec) > 0):
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=5e-16)
    else:
        np.testing.assert_array_equal(result, np.zeros_like(vec))
