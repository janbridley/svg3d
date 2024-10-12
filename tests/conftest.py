import numpy as np
import pytest

MAX_N = 12
SEED = 13579


@pytest.fixture
def rng():
    return np.random.default_rng(seed=SEED)


@pytest.fixture
def random_ragged_array(rng):
    """Create a ragged array with random length"""

    def generate_ragged_array(n=MAX_N):
        num_arrays = rng.integers(1, n + 1)
        arrays = [
            rng.uniform(size=(rng.integers(1, n + 1), 3)) for _ in range(num_arrays)
        ]
        return arrays

    return generate_ragged_array()


def filter_invalid_arrays(arr, min_value):
    """Return False for arrays containing nonzero numbers smaller than a min_value."""
    # if np.any((np.abs(arr) < min_value) & (arr != np.zeros_like(arr))):
    #     print(arr)
    return not np.any((np.abs(arr) < min_value) & (arr != np.zeros_like(arr)))
