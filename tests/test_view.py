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
    """
    Extensive configuration is required for this test to remove extreme degenerate
    hypothesis cases. For all 'reasonable' inputs, the results are equal to machine
    precision.
    """
    pos_object, pos_camera = pos

    npt.assert_allclose(
        get_lookat_matrix(pos_object, pos_camera, vec_up),
        pyrr.matrix44.create_look_at(eye=pos_camera, target=pos_object, up=vec_up),
        atol=PRECISION,
    )


@given(
    z_near=floats(MIN_VALID_DISTANCE, 100, exclude_min=True),
    z_far_distance=floats(MIN_VALID_DISTANCE, 100, exclude_min=True),
    fov_y=floats(MIN_VALID_DISTANCE, 180, exclude_min=True, exclude_max=True),
    aspect=floats(1 / 100, 100),
)
def test_get_projection_matrix(z_near, z_far_distance, fov_y, aspect):
    z_far = z_near + z_far_distance
    npt.assert_allclose(
        get_projection_matrix(z_near, z_far, fov_y, aspect),
        pyrr.matrix44.create_perspective_projection_matrix(
            fov_y, aspect, z_near, z_far
        ),
        atol=PRECISION,
    )
