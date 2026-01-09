import numpy as np


def _stable_normalize(vec: np.ndarray, min_nonzero_value=0.0):
    """Normalize a vector or set of vectors in a numerically stable way.

    Allows for an optional tolerance below which the input will be treated as the 0 vec.
    """
    vec = np.asanyarray(vec)
    max_coeff = np.max(np.abs(vec), axis=-1, keepdims=True)

    # Create a divisor array that scales our vector such that each row is in [-1, 1]
    is_nonzero = max_coeff > min_nonzero_value
    safe_max = np.where(is_nonzero, max_coeff, 1.0)

    # pre-scale the vector to handle overflow & underflow
    scaled = vec / safe_max

    scaled_norm = np.linalg.norm(scaled, axis=-1, keepdims=True)

    # scaled_norm will be >= 1.0 where is_nonzero is True
    # For safety, avoid division by zero in the non-selected branches
    safe_norm = np.where(is_nonzero, scaled_norm, 1.0)

    normalized = scaled / safe_norm
    return np.where(is_nonzero, normalized, 0.0)
