import numpy as np
import numpy.testing as npt
import pyrr
import pytest
from conftest import filter_invalid_arrays
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats

from svg3d import get_lookat_matrix, get_projection_matrix

MIN_VALID_DISTANCE = 1e-12
PRECISION = 1e-12  # Small deviations are OK given the rigor of hypothesis testing


@pytest.mark.filterwarnings("ignore: invalid value encountered in divide")
@pytest.mark.parametrize(
    "vec_up", [(0, 1, 0), (1, 0, 0), (0, 0, 1), (123.0, -45, 0.0001)]
)
@given(
    pos=arrays(np.float64, (2, 3), elements=floats(-100, 100)).filter(
        lambda x: (
            (np.linalg.norm(x[0] - x[1]) > MIN_VALID_DISTANCE)
            and filter_invalid_arrays(x, PRECISION)
        )
    ),
)
def test_get_lookat_matrix(pos, vec_up):
    pos_object, pos_camera = pos

    npt.assert_allclose(
        get_lookat_matrix(pos_object, pos_camera, vec_up),
        pyrr.matrix44.create_look_at(eye=pos_camera, target=pos_object, up=vec_up),
        atol=PRECISION,
    )


def test_get_projection_matrix():
    assert get_projection_matrix
