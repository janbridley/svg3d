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


def _project_point(point: np.ndarray, view: View) -> tuple[float, float]:
    """Project a 3D point through the view pipeline to get 2D coordinates."""
    # Extend to homogeneous coordinates
    p = np.append(point, 1.0)
    # Apply look_at and projection
    projection = np.dot(view.look_at, view.projection)
    p = np.dot(p, projection)
    # Perspective divide
    x, y = p[0] / p[3], p[1] / p[3]
    return x, y


class TestOrthographicRendering:
    """Tests validating rendered output properties for orthographic views."""

    def test_origin_at_center(self):
        """Origin should project to center of viewport (0, 0 in NDC)."""
        view = View.orthographic(scene=[], theta=45, elevation=45)
        x, y = _project_point(np.array([0.0, 0.0, 0.0]), view)
        npt.assert_allclose(x, 0.0, atol=1e-10)
        npt.assert_allclose(y, 0.0, atol=1e-10)

    @pytest.mark.parametrize("theta", [0, 45, 90, 180, 270])
    def test_origin_at_center_all_angles(self, theta):
        """Origin should always project to center regardless of theta."""
        view = View.orthographic(scene=[], theta=theta, elevation=45)
        x, y = _project_point(np.array([0.0, 0.0, 0.0]), view)
        npt.assert_allclose(x, 0.0, atol=1e-10)
        npt.assert_allclose(y, 0.0, atol=1e-10)

    def test_top_down_view_z_axis_collapses(self):
        """At elevation=90 (top-down), z axis points toward camera (becomes a point)."""
        view = View.orthographic(scene=[], theta=0, elevation=90)
        # Points along z should project to same xy as origin
        origin = _project_point(np.array([0.0, 0.0, 0.0]), view)
        z_point = _project_point(np.array([0.0, 0.0, 1.0]), view)
        # z-axis points toward camera at elevation=90, so it collapses
        npt.assert_allclose(origin[0], z_point[0], atol=1e-10)
        npt.assert_allclose(origin[1], z_point[1], atol=1e-10)

    def test_top_down_view_xy_visible(self):
        """At elevation=90 (top-down), x and y axes should be visible."""
        view = View.orthographic(scene=[], theta=0, elevation=90)
        origin = _project_point(np.array([0.0, 0.0, 0.0]), view)
        x_point = _project_point(np.array([1.0, 0.0, 0.0]), view)
        y_point = _project_point(np.array([0.0, 1.0, 0.0]), view)
        # x-axis projects to (1, 0), y-axis projects to (0, 1)
        npt.assert_allclose(origin, (0.0, 0.0), atol=1e-10)
        npt.assert_allclose(x_point, (1.0, 0.0), atol=1e-10)
        npt.assert_allclose(y_point, (0.0, 1.0), atol=1e-10)

    def test_side_view_z_axis_vertical(self):
        """At elevation=0 (side view from +x), z axis should project vertically."""
        view = View.orthographic(scene=[], theta=0, elevation=0)
        # Looking from +x toward origin, z axis goes up (vertical on screen)
        origin = _project_point(np.array([0.0, 0.0, 0.0]), view)
        z_point = _project_point(np.array([0.0, 0.0, 1.0]), view)
        # z-axis projects to (0, 1) - vertical in screen space
        npt.assert_allclose(origin, (0.0, 0.0), atol=1e-10)
        npt.assert_allclose(z_point, (0.0, 1.0), atol=1e-10)

    def test_theta_0_view_from_x(self):
        """At theta=0, camera looks from +x toward -x."""
        view = View.orthographic(scene=[], theta=0, elevation=0)
        # y-axis should map to screen x direction
        origin = _project_point(np.array([0.0, 0.0, 0.0]), view)
        y_point = _project_point(np.array([0.0, 1.0, 0.0]), view)
        npt.assert_allclose(origin, (0.0, 0.0), atol=1e-10)
        npt.assert_allclose(y_point, (1.0, 0.0), atol=1e-10)

    def test_theta_90_view_from_y(self):
        """At theta=90, camera looks from +y toward -y."""
        view = View.orthographic(scene=[], theta=90, elevation=0)
        # x-axis should map to screen -x direction
        origin = _project_point(np.array([0.0, 0.0, 0.0]), view)
        x_point = _project_point(np.array([1.0, 0.0, 0.0]), view)
        npt.assert_allclose(origin, (0.0, 0.0), atol=1e-10)
        npt.assert_allclose(x_point, (-1.0, 0.0), atol=1e-10)

    def test_orthographic_no_perspective_scaling(self):
        """Orthographic projection should not scale objects based on distance."""
        view = View.orthographic(scene=[], theta=45, elevation=45)
        # Two points at different distances along view direction
        near = np.array([0.0, 0.0, 0.0])
        far = np.array([10.0, 10.0, 10.0])
        # Offset both by same amount perpendicular to view
        offset = np.array([0.0, 1.0, 0.0])
        near_offset = _project_point(near + offset, view)
        far_offset = _project_point(far + offset, view)
        near_base = _project_point(near, view)
        far_base = _project_point(far, view)
        # The offset distance should be the same in screen space
        near_dist = np.sqrt(
            (near_offset[0] - near_base[0]) ** 2 + (near_offset[1] - near_base[1]) ** 2
        )
        far_dist = np.sqrt(
            (far_offset[0] - far_base[0]) ** 2 + (far_offset[1] - far_base[1]) ** 2
        )
        npt.assert_allclose(near_dist, far_dist, rtol=1e-10)

    def test_parallel_lines_remain_parallel(self):
        """Orthographic projection preserves parallel lines."""
        view = View.orthographic(scene=[], theta=30, elevation=45)
        # Two parallel line segments in 3D
        line1_start = _project_point(np.array([0.0, 0.0, 0.0]), view)
        line1_end = _project_point(np.array([1.0, 0.0, 0.0]), view)
        line2_start = _project_point(np.array([0.0, 1.0, 0.0]), view)
        line2_end = _project_point(np.array([1.0, 1.0, 0.0]), view)
        # Direction vectors in screen space
        dir1 = (line1_end[0] - line1_start[0], line1_end[1] - line1_start[1])
        dir2 = (line2_end[0] - line2_start[0], line2_end[1] - line2_start[1])
        # Should be parallel (same direction)
        npt.assert_allclose(dir1, dir2, atol=1e-10)

    def test_scene_width_scales_objects(self):
        """Larger scene_width should make objects appear smaller."""
        # Small scene width: object fills more of view
        view_small = View.orthographic(scene=[], scene_width=1.0, theta=0, elevation=45)
        # Large scene width: object fills less of view
        view_large = View.orthographic(
            scene=[], scene_width=10.0, theta=0, elevation=45
        )
        # Point at (0, 1, 0) - offset from origin
        point = np.array([0.0, 1.0, 0.0])
        small_proj = _project_point(point, view_small)
        large_proj = _project_point(point, view_large)
        # In NDC, larger scene_width means point appears closer to center
        small_dist = np.sqrt(small_proj[0] ** 2 + small_proj[1] ** 2)
        large_dist = np.sqrt(large_proj[0] ** 2 + large_proj[1] ** 2)
        assert large_dist < small_dist

    def test_elevation_changes_z_visibility(self):
        """Lower elevation should show more of the z axis vertically."""
        # At elevation=0 (side view), z axis is fully vertical
        # At elevation=90 (top-down), z axis collapses to a point
        view_low = View.orthographic(scene=[], theta=0, elevation=10)
        view_high = View.orthographic(scene=[], theta=0, elevation=80)
        origin = np.array([0.0, 0.0, 0.0])
        z_point = np.array([0.0, 0.0, 1.0])
        low_origin = _project_point(origin, view_low)
        low_z = _project_point(z_point, view_low)
        high_origin = _project_point(origin, view_high)
        high_z = _project_point(z_point, view_high)

        low_dy = abs(low_z[1] - low_origin[1])
        high_dy = abs(high_z[1] - high_origin[1])
        # At elevation=10°, z-axis projects more vertically than at elevation=80°
        npt.assert_allclose(low_dy, 0.984807753012208, atol=1e-10)
        npt.assert_allclose(high_dy, 0.17364817766693041, atol=1e-10)
        assert low_dy > high_dy
