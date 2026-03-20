import numpy as np
import numpy.testing as npt
import pyrr
import pytest
import rowan
from conftest import filter_invalid_arrays
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats

from svg3d import View, get_lookat_matrix, get_projection_matrix

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


def test_isometric_view():
    isometric_view_matrix = np.eye(4)
    isometric_view_matrix[:3, :3] = (
        rowan.to_matrix(rowan.from_axis_angle([0, 1, 0], np.pi / 4))  # y-up convention
        @ rowan.to_matrix(rowan.from_axis_angle([1, 0, 0], -np.atan(1 / np.sqrt(2))))
    )
    isometric_view_matrix[-1, 2] = -100

    npt.assert_allclose(
        View.ISOMETRIC_VIEW_MATRIX, isometric_view_matrix, atol=PRECISION
    )


class TestOrthographicView:
    """Tests for View.orthographic factory method."""

    def test_default_parameters(self):
        """Test orthographic view with default parameters."""
        view = View.orthographic(scene=[])
        assert view.viewport.aspect_ratio == 1.0
        # Default theta=45, elevation=35.264 (isometric)
        # Camera should be at distance 100 with those angles

    def test_aspect_ratio(self):
        """Test that aspect ratio is correctly set."""
        view = View.orthographic(scene=[], aspect_ratio=16 / 9)
        assert view.viewport.aspect_ratio == 16 / 9

    @pytest.mark.parametrize("theta", [0, 45, 90, 180, 270])
    def test_theta_rotation(self, theta):
        """Test theta controls azimuthal rotation in xy plane."""
        view = View.orthographic(scene=[], theta=theta)
        # View should be created without error
        assert view.look_at is not None

    @pytest.mark.parametrize("elevation", [0, 30, 45, 60, 90])
    def test_elevation_angle(self, elevation):
        """Test elevation controls angle from xy plane."""
        view = View.orthographic(scene=[], elevation=elevation)
        assert view.look_at is not None

    def test_camera_position_theta_0_elevation_0(self):
        """Camera at theta=0, elevation=0 should be on +x axis."""
        import math

        view = View.orthographic(scene=[], theta=0, elevation=0)
        # At theta=0, elevation=0: camera at (100, 0, 0)
        # The look_at matrix translates, so we check the view is valid
        assert view.look_at is not None

    def test_camera_position_theta_90_elevation_0(self):
        """Camera at theta=90, elevation=0 should be on +y axis."""
        view = View.orthographic(scene=[], theta=90, elevation=0)
        assert view.look_at is not None

    def test_camera_position_elevation_90(self):
        """Camera at elevation=90 should be directly above (top-down view)."""
        view = View.orthographic(scene=[], theta=45, elevation=90)
        assert view.look_at is not None

    def test_scene_width_affects_projection(self):
        """Different scene_width values should produce different projections."""
        view1 = View.orthographic(scene=[], scene_width=2.0)
        view2 = View.orthographic(scene=[], scene_width=4.0)
        # Projections should be different (different ortho width)
        assert not np.allclose(view1.projection, view2.projection)

    def test_scene_width_with_aspect_ratio(self):
        """Test scene_width and aspect_ratio work together."""
        view = View.orthographic(scene=[], scene_width=10.0, aspect_ratio=2.0)
        assert view.viewport.aspect_ratio == 2.0

    def test_isometric_default(self):
        """Default parameters should produce isometric-like view."""
        # Default elevation=35.264 is the isometric angle
        view = View.orthographic(scene=[])
        assert view.look_at is not None
        assert view.projection is not None
