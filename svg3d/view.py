"""Define OpenGL-style views and viewports for scene rendering."""

import math
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np

from .svg3d import Mesh


def get_lookat_matrix(
    pos_object: np.ndarray,
    pos_camera: np.ndarray,
    vec_up: np.ndarray | tuple = (0.0, 1.0, 0.0),
):
    """Get the "look at" or view matrix for our system.

    This matrix moves the world such that the camera is at the origin and rotates the
    world such that the z-axis of the camera is the mathematical z axis.


    Parameters
    ----------
    pos_object : :math:`(3,)` :class:`numpy.ndarray`
        Position of the object we are looking at. "at" in openGL vernacular.
    pos_camera : :math:`(3,)` :class:`numpy.ndarray`
        Position of the camera. "eye" in openGL vernacular.
    vec_up : :math:`(3,)` :class:`numpy.ndarray`: | tuple, optional
        Vector describing the height of the camera. "up" in openGL vernacular.
        Default value: (0.0, 1.0, 0.0)


    .. seealso:: Calculating a Lookat Matrix:

        https://stackoverflow.com/questions/349050/calculating-a-lookat-matrix/6802424#6802424

    .. seealso:: Understanding Lookat Matrices:

        https://medium.com/@carmencincotti/lets-look-at-magic-lookat-matrices-c77e53ebdf78
    """
    # First, shift the world such that the camera is at the origin
    m_camera_translate = np.eye(4)
    m_camera_translate[-1, :3] -= pos_camera

    # Now, rotate the vector from the camera position to the object position such that
    # it lines up with the z axis.

    # Compute the x axis of our original coordinates along the vector [camera - pos]
    axis_z = np.asarray(pos_camera, dtype=np.float64) - pos_object
    axis_z /= np.linalg.norm(axis_z)  # "forward" axis in openGL terms

    # Compute the y ("forward") axis of our original coordinate system. This is
    # perpendicular to axis_z and any arbitrary vector in the plane formed by z and y
    axis_x = np.cross(vec_up, axis_z)
    axis_x /= np.linalg.norm(axis_x)  # "right" axis in openGL terms

    axis_y = np.cross(axis_z, axis_x)  # "up" axis in openGL terms

    m_camera_rotate = np.eye(4)
    m_camera_rotate[:3, :3] = [axis_x, axis_y, axis_z]

    return m_camera_translate @ (m_camera_rotate.T)


def get_projection_matrix(
    z_near: float, z_far: float, fov_y: float, aspect: float = 1.0
):
    """Get a projection matrix from parameters of the provided view frustum.

    z_near and z_far are the distances to the tip and base of the frustum, respectively.
    fov_y describes the opening angle of the base, and aspect describes the relationship
    between the y opening angle and the x. Objects that lie outside the view frustum are
    culled and wil not be rendered into the scene.

    .. # TODO: include image of frustum view

    Parameters
    ----------
    z_near : float
        Distance to the near clipping plane. Must be greater than zero.
    z_far : float
        Distance to the far clipping plane. Must be greater than z_near.
    fov_y : float
        Field of view angle along the y direction, in degrees.
    aspect : float, optional
        Ratio of field of view angle in the y direction to field of view angle in x.
        Default value: 1.0


    .. seealso:: OpenGL Reference:

        https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml

    .. seealso:: Understanding Projection Matrices:

        http://www.songho.ca/opengl/gl_projectionmatrix.html

    """
    f = 1 / math.tan(math.radians(fov_y) / 2)
    m_projection = np.zeros([4, 4])

    m_projection[[0, 1, -1], [0, 1, 2]] = f / aspect, f, -1
    m_projection[2, [2, 3]] = (
        (z_near + z_far) / (z_near - z_far),
        (2 * z_near * z_far) / (z_near - z_far),
    )
    return m_projection.T


def get_orthographic_matrix(
    width: float,
    height: float,
    z_near: float = 1.0,
    z_far: float = 200.0,
) -> np.ndarray:
    """Get an orthographic (parallel) projection matrix.

    Unlike perspective projection, orthographic projection does not
    create depth-based scaling - objects remain the same size regardless
    of their distance from the camera.

    This creates a symmetric orthographic projection centered on the view
    direction, equivalent to glOrtho with left=-width/2, right=width/2,
    bottom=-height/2, top=height/2.

    Parameters
    ----------
    width : float
        Width of the view volume.
    height : float
        Height of the view volume.
    z_near : float
        Distance to the near clipping plane. Default: 1.0
    z_far : float
        Distance to the far clipping plane. Default: 200.0

    Returns
    -------
    np.ndarray
        4x4 orthographic projection matrix.

    References
    ----------
    .. [1] https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/orthographic-projection-matrix.html
    .. [2] https://songho.ca/opengl/gl_projectionmatrix.html
    """
    m = np.zeros([4, 4])
    m[0, 0] = 2 / width
    m[1, 1] = 2 / height
    m[2, 2] = -2 / (z_far - z_near)
    m[2, 3] = -(z_far + z_near) / (z_far - z_near)
    m[3, 3] = 1
    return m.T


class Viewport(NamedTuple):
    """A :obj:`~.Viewport` controls the visible area in a rendered SVG.

    This is a convience wrapper around the svgwrite :obj:`~svgwrite.mixins.Viewbox`
    classes with a simplified interface.
    """

    minx: float = -0.5
    """Left border of the viewport."""
    miny: float = -0.5
    """Right border of the viewport."""
    width: float = 1.0
    """Width of the viewport."""
    height: float = 1.0
    """Height of the viewport."""

    @property
    def aspect_ratio(self) -> float:
        """The aspect ratio (width/height) of the viewport."""
        return self.width / self.height

    def get_size(self, base_height: int = 512) -> tuple[int, int]:
        """Get a recommended render size in pixels.

        Parameters
        ----------
        base_height : int
            Base height in pixels. Width is derived from aspect ratio.

        Returns
        -------
        tuple[int, int]
            (width, height) in pixels matching the viewport aspect ratio.
        """
        return (int(base_height * self.aspect_ratio), base_height)

    @classmethod
    def from_aspect(cls, aspect_ratio: float):
        """Create a :obj:`~.Viewport` with the given aspect ratio."""
        return cls(-aspect_ratio / 2.0, -0.5, aspect_ratio, 1.0)

    @classmethod
    def from_string(cls, string_to_parse: str):
        """Create a :obj:`~.Viewport` from a space-delimited string of floats."""
        args = [float(f) for f in string_to_parse.split()]
        return cls(*args)


class View:
    def __init__(
        self,
        look_at: np.ndarray,
        projection: np.ndarray,
        scene: tuple[Mesh] | list[Mesh],
        viewport=None,
    ):
        self._look_at = look_at
        self._projection = projection
        self._scene = scene
        self._viewport = viewport if viewport is not None else Viewport()

    DEFAULT_OBJECT_POSITION = np.zeros(3)
    """Classmethods for this object center their view on the origin by default."""

    ISOMETRIC_VIEW_MATRIX = [
        [np.sqrt(3), -1, np.sqrt(2), 0],
        [0, 2, np.sqrt(2), 0],
        [-np.sqrt(3), -1, np.sqrt(2), 0],
        [0, 0, -100 * np.sqrt(6), np.sqrt(6)],
    ] / np.sqrt(6)  # TODO: no-undoc-members, don't want to expose this

    @property
    def look_at(self):
        """:math:`(4,4)` :class:`numpy.ndarray`: The openGL-style lookAt matrix.

        .. TODO: add links to openGL docs, explain transpose if required.
        """
        return self._look_at

    @look_at.setter
    def look_at(self, look_at: np.ndarray):
        self._look_at = look_at

    @property
    def projection(self):
        """:math:`(4,4)` :class:`numpy.ndarray`: The openGL-style projection matrix.

        .. TODO: add links to openGL docs, explain transpose if required.
        """
        return self._projection

    @projection.setter
    def projection(self, projection: np.ndarray):
        self._projection = projection

    @property
    def scene(self):
        """Iterable[Mesh] : Get or set the list of :obj:`~.Mesh` objects to render."""
        return self._scene

    @scene.setter
    def scene(self, scene: tuple[Mesh] | list[Mesh]):
        self._scene = scene

    @property
    def viewport(self):
        """Viewport: Get or set the system's :obj:`~.Viewport`."""
        return self._viewport

    @viewport.setter
    def viewport(self, viewport: Viewport):
        self._viewport = viewport

    @classmethod
    def from_look_at_and_projection(
        cls,
        look_at: np.ndarray,
        projection: np.ndarray,
        scene: Iterable[Mesh],
    ):
        """Create a new :obj:`~.View` from a lookAt and projection matrix.


        .. TODO: Describe how these are composed, give matrix equations
        """
        msg = "Both look_at and projection must have size (4,4)."
        assert look_at.shape == (4, 4) and projection.shape == (4, 4), msg
        return cls(
            look_at,
            projection,
            scene,
        )

    @classmethod
    def isometric(cls, scene, scene_width: float = 2.0, aspect_ratio: float = 1.0):
        """Create a :obj:`~.View` based on an isometric projection.

        In an isometric projection, the scale along each coordinate axis is identical.
        All three axes are equally foreshortened by a factor of √(2/3) ≈ 0.816.
        This is a parallel projection method, meaning that objects remain the same size
        regardless of their position from the camera.

        The isometric view uses:
        - theta = 45° (azimuthal rotation in xy plane)
        - elevation = 35.264° (arcsin(1/√3), angle from xy plane)

        Parameters
        ----------
        scene : list[Mesh]
            An iterable of mesh objects to view.
        scene_width : float
            Width of the view volume. Default: 2.0
        aspect_ratio : float
            Width/height ratio of the viewport. Default: 1.0

        Returns
        -------
        View
            A View with true orthographic isometric projection.
        """
        # Isometric: all axes equally foreshortened
        # theta = 45°, elevation = arcsin(1/√3) ≈ 35.264°
        return cls.orthographic(
            scene=scene,
            scene_width=scene_width,
            aspect_ratio=aspect_ratio,
            theta=45.0,
            elevation=35.264389682754654,  # arcsin(1/√3) in degrees
        )

    @classmethod
    def orthographic(
        cls,
        scene: list,
        scene_width: float = 2.0,
        aspect_ratio: float = 1.0,
        theta: float = 45.0,
        elevation: float = 35.264,
        z_near: float = 1.0,
        z_far: float = 200.0,
    ):
        """Create a View with true orthographic projection.

        With orthographic projection, objects remain the same size regardless of
        their distance from the camera - there is no perspective distortion.

        Parameters
        ----------
        scene : list[Mesh]
            An iterable of mesh objects to view.
        scene_width : float
            Width of the view volume. Default: 2.0
        aspect_ratio : float
            Width/height ratio of the viewport. Default: 1.0
        theta : float
            Azimuthal angle in degrees (rotation in xy plane).
            0° = viewing from +x direction, 90° = from +y. Default: 45.0
        elevation : float
            Angle from xy plane toward z in degrees.
            0° = horizontal view, 90° = top-down view. Default: 35.264 (isometric)
        z_near : float
            Distance to the near clipping plane. Default: 1.0
        z_far : float
            Distance to the far clipping plane. Default: 200.0

        Returns
        -------
        View
            A View with orthographic projection.
        """
        # Convert to radians
        theta_rad = math.radians(theta)
        elevation_rad = math.radians(elevation)

        # Camera position using spherical coordinates
        # elevation: angle from xy plane (0=horizontal, 90=top-down)
        distance = 100.0
        x = distance * math.cos(elevation_rad) * math.cos(theta_rad)
        y = distance * math.cos(elevation_rad) * math.sin(theta_rad)
        z = distance * math.sin(elevation_rad)

        pos_camera = np.array([x, y, z])
        pos_object = np.zeros(3)

        # Up vector: for elevation < 90, use z-up rotated by theta
        # For near top-down views, fall back to y-up
        if abs(elevation) < 89.0:
            vec_up = np.array([0.0, 0.0, 1.0])
        else:
            vec_up = np.array(
                [
                    -math.sin(theta_rad),
                    math.cos(theta_rad),
                    0.0,
                ]
            )

        look_at = get_lookat_matrix(pos_object, pos_camera, vec_up=vec_up)
        viewport = Viewport.from_aspect(aspect_ratio)
        height = scene_width / aspect_ratio

        return cls(
            look_at=look_at,
            projection=get_orthographic_matrix(scene_width, height, z_near, z_far),
            scene=scene,
            viewport=viewport,
        )

    @classmethod
    def dimetric(cls, scene, scene_width: float = 2.0, aspect_ratio: float = 1.0):
        """Create a :obj:`~.View` based on a dimetric projection.

        In a dimetric projection, the scale along two out of three axes is identical.
        This strikes a balance between the simplicity and interpretability of isometric
        projections and the improved sense of realism afforded by trimetric projections.

        This implementation uses a common dimetric configuration where the x and y axes
        have equal foreshortening (~0.94) while the z axis has different foreshortening
        (~0.47). This is achieved with:
        - theta = 45° (azimuthal rotation in xy plane)
        - elevation = 20.705° (arcsin(1/√8), angle from xy plane)

        Parameters
        ----------
        scene : list[Mesh]
            An iterable of mesh objects to view.
        scene_width : float
            Width of the view volume. Default: 2.0
        aspect_ratio : float
            Width/height ratio of the viewport. Default: 1.0

        Returns
        -------
        View
            A View with true orthographic dimetric projection.

        References
        ----------
        .. [1] https://www.math.tu-cottbus.de/~klempp/Folie12.pdf
        """
        # Dimetric: two axes equally foreshortened
        # theta = 45°, elevation = arcsin(1/√8) ≈ 20.705°
        # This gives x:y:z foreshortening of ~0.94:0.94:0.47
        return cls.orthographic(
            scene=scene,
            scene_width=scene_width,
            aspect_ratio=aspect_ratio,
            theta=45.0,
            elevation=20.704811054635432,  # arcsin(1/√8) in degrees
        )

    @classmethod
    def trimetric(cls, scene, scene_width: float = 2.0, aspect_ratio: float = 1.0):
        """Create a :obj:`~.View` based on a trimetric projection.

        In a trimetric projection, each axis is scaled independently. This results in a
        more "natural" scene than isometric and dimetric views, as the foreshortening
        of each axis provides a sense of depth to the scene.

        This is a parallel projection method, meaning that objects remain the same size
        regardless of their position from the camera.

        This implementation uses angles chosen to give distinct foreshortening for all
        three axes:
        - theta = 30° (azimuthal rotation in xy plane)
        - elevation = 25° (angle from xy plane)

        Parameters
        ----------
        scene : list[Mesh]
            An iterable of mesh objects to view.
        scene_width : float
            Width of the view volume. Default: 2.0
        aspect_ratio : float
            Width/height ratio of the viewport. Default: 1.0

        Returns
        -------
        View
            A View with true orthographic trimetric projection.
        """
        # Trimetric: all three axes have different foreshortening
        # Using theta=30°, elevation=25° gives distinct foreshortening for each axis
        return cls.orthographic(
            scene=scene,
            scene_width=scene_width,
            aspect_ratio=aspect_ratio,
            theta=30.0,
            elevation=25.0,
        )


# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
