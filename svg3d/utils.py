import numpy as np


def _stable_normalize(vec: np.ndarray, min_nonzero_value=0.0):
    """Normalize a vector in a numerically stable way.

    Allows for an optional tolerance below which the input will be treated as the 0 vec.
    """
    max_coeff = np.abs(vec).max()
    if max_coeff <= min_nonzero_value:
        return np.zeros_like(vec)

    # pre-scale the vector to handle overflow & underflow
    scaled = vec / max_coeff

    scaled_norm = np.linalg.norm(scaled)
    return scaled / scaled_norm
