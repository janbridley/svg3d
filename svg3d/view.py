import math
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


    Args:
        pos_object (np.ndarray):
          (N,3) position of the object we are looking at. "at" in openGL vernacular.
        pos_camera (np.ndarray):
          (N,3) position of the camera. "eye" in openGL vernacular.
        vec_up (np.ndarray | tuple, optional):
          (N,3) vector describing the height of the camera. "up" in openGL vernacular.
          Default value: (0,1,0)

    .. seealso:

        https://stackoverflow.com/questions/349050/calculating-a-lookat-matrix/6802424#6802424

    .. seealso:

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
    """Get a projection matrix from frustum parameters.

    z_near and z_far are the distances to the tip and base of the frustum, respectively.
    fov_y describes the opening angle of the base, and aspect describes the relationship
    between the y opening angle and the x.

    Args:
        z_near (float): Distance to the near clipping plane. Must be greater than zero.
        z_far (float): Distance to the far clipping plane. Must be greater than z_near.
        fov_y (float): Field of view angle along the y direction, in degrees.
        aspect (float, optional):
          Ratio of field of view angle in the y direction to field of view angle in x.
          Default value: 1.0

    .. seealso:

        https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml

    .. seealso:

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

    ISOMETRIC_VIEW_MATRIX = [
        [np.sqrt(3), -1, np.sqrt(2), 0],
        [0, 2, np.sqrt(2), 0],
        [-np.sqrt(3), -1, np.sqrt(2), 0],
        [0, 0, -100 * np.sqrt(6), np.sqrt(6)],
    ] / np.sqrt(6)

    @property
    def look_at(self):
        return self._look_at

    @look_at.setter
    def look_at(self, look_at: np.ndarray):
        self._look_at = look_at

    @property
    def projection(self):
        return self._projection

    @projection.setter
    def projection(self, projection: np.ndarray):
        self._projection = projection

    @property
    def scene(self):
        return self._scene

    @scene.setter
    def scene(self, scene: tuple[Mesh] | list[Mesh]):
        self._scene = scene

    @property
    def viewport(self):
        return self._viewport

    @viewport.setter
    def viewport(self, viewport):
        self._viewport = viewport

    @classmethod
    def from_look_at_and_projection(
        cls,
        look_at: np.ndarray,
        projection: np.ndarray,
        scene: tuple[Mesh],
    ):
        assert look_at.shape == (4, 4) and projection.shape == (4, 4)
        return cls(
            look_at,
            projection,
            scene,
        )

    @classmethod
    def isometric(cls, scene, fov: float = 1.0, distance: float = 100.0):
        # Equivalent to a 45 degree rotation about the X axis and an atan(1/sqrt(2))
        # degree rotation about the z axis
        isometric_view = cls.ISOMETRIC_VIEW_MATRIX
        isometric_view[-1, 2] = -distance

        return cls(
            look_at=isometric_view,
            projection=get_projection_matrix(z_near=1.0, z_far=200.0, fov_y=fov),
            scene=scene,
        )

    @classmethod
    def dimetric(cls, scene, fov: float = 1.0, distance: float = 100.0):
        # TODO: reimplement as https://faculty.sites.iastate.edu/jia/files/inline-files/projection-classify.pdf
        camera_position = np.array([8, 8, 21]) / math.sqrt(569) * distance
        return cls(
            look_at=get_lookat_matrix(
                pos_object=cls.DEFAULT_OBJECT_POSITION, pos_camera=camera_position
            ),
            projection=get_projection_matrix(z_near=1.0, z_far=200.0, fov_y=fov),
            scene=scene,
        )

    @classmethod
    def trimetric(cls, scene, fov: float = 1.0, distance: float = 100.0):
        camera_position = np.array([1 / 7, 1 / 14, 3 / 14]) * math.sqrt(14) * distance
        return cls(
            look_at=get_lookat_matrix(
                pos_object=cls.DEFAULT_OBJECT_POSITION, pos_camera=camera_position
            ),
            projection=get_projection_matrix(z_near=1.0, z_far=200.0, fov_y=fov),
            scene=scene,
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
