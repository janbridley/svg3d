import numpy as np
import pytest
from hypothesis import given, settings
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


@pytest.mark.parametrize("shape", [(1,), (2,), (3,), (5,), (99, 3)])
def test_stable_normalize_zero_vector(shape):
    """Zero vector should return zero vector for various dimensions."""
    vec = np.zeros(shape)
    result = _stable_normalize(vec)
    np.testing.assert_array_equal(result, vec)


@settings(max_examples=1000)
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


@settings(max_examples=1000)
@given(
    np_strategies.arrays(
        dtype=np.float64,
        shape=np_strategies.array_shapes(min_dims=2, max_dims=2, min_side=1).map(
            lambda s: (s[0], 3)
        ),
        elements=st.floats(allow_nan=False, allow_infinity=False),
    )
)
def test_stable_normalize_n3_vector(vecs):
    """Result should have unit length for each non-zero row in an array."""
    result = _stable_normalize(vecs)
    norms = np.linalg.norm(result, axis=-1)
    input_max = np.max(np.abs(vecs), axis=-1)

    is_nonzero = input_max > 0
    np.testing.assert_allclose(norms[is_nonzero], 1.0, atol=5e-16)
    np.testing.assert_array_equal(result[~is_nonzero], 0.0)
