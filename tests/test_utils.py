from svg3d import _pad_arrays


def test_pad_arrays(random_ragged_array):
    subarray_count = len(random_ragged_array)
    subarray_max_len = max(len(arr) for arr in random_ragged_array)

    assert _pad_arrays(random_ragged_array).shape == (
        subarray_count,
        subarray_max_len,
        3,
    )
