# svg3d :: https://prideout.net/blog/svg_wireframes/
# Single-file Python library for generating 3D wireframes in SVG format.
# Adapted from https://prideout.net/blog/svg_wireframes/
# Copyright (c) 2019 Philip Rideout. Modified 2024 by Jenna Bradley.
# Distributed under the MIT License, see bottom of file.


import math
import warnings
from typing import TYPE_CHECKING, Callable, NamedTuple

import numpy as np
import svgwrite

if TYPE_CHECKING:
    import coxeter


def _pad_arrays(arrays):
    # Find the length of the longest array
    max_length = max(len(arr) for arr in arrays)

    # Pad each array to the length of the longest array
    padded_array = [
        np.concatenate((arr, np.full((max_length - len(arr), 3), arr[0])), axis=0)
        for arr in arrays
    ]
    return np.array(padded_array)


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
    minx: float = -0.5
    miny: float = -0.5
    width: float = 1.0
    height: float = 1.0

    @classmethod
    def from_aspect(cls, aspect_ratio: float):
        return cls(-aspect_ratio / 2.0, -0.5, aspect_ratio, 1.0)

    @classmethod
    def from_string(cls, string_to_parse: str):
        args = [float(f) for f in string_to_parse.split()]
        return cls(*args)


class Mesh:
    def __init__(
        self,
        faces: list[np.ndarray],
        shader: Callable[[int, float], dict] | None = None,
        style: dict | None = None,
        circle_radius: float = 0.0,
    ):
        self._faces = _pad_arrays(faces)
        self._shader = shader
        self._style = style
        self._circle_radius = circle_radius

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, faces):
        self._faces = faces

    @property
    def shader(self):
        return self._shader

    @shader.setter
    def shader(self, shader):
        self._shader = shader

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, shader):
        self._shader = shader

    @property
    def circle_radius(self):
        return self._circle_radius

    @circle_radius.setter
    def circle_radius(self, circle_radius):
        self._circle_radius = circle_radius

    @property
    def normals(self):
        face_simplices = self.faces[:, :3]

        # Convert each simplex (3 points) into two edge vectors (each 2 points)
        # These will be an array of [N, (0-1,1-2)=2, 3] vertices
        face_edge_vectors = np.diff(face_simplices, axis=1)

        # The LSP is unhappy, but this is correct. Each face has exactly 2 edge vectors
        normals = np.cross(*np.split(face_edge_vectors, 2, axis=1)).squeeze()

        return normals / np.linalg.norm(normals)  # Return normalized

    @classmethod
    def from_coxeter(
        cls,
        poly: "coxeter.shapes.ConvexPolyhedron",
        shader=None,
        style=None,
    ):  # noqa: F821
        return cls(
            faces=[poly.vertices[face] for face in poly.faces],
            shader=shader,
            style=style,
        )


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
        [np.sqrt(3), 0, -np.sqrt(3), 0],
        [1, 2, 1, 0],
        [np.sqrt(2), -np.sqrt(2), np.sqrt(2), 0],
        [0, 0, 0, np.sqrt(6)],
    ] / np.sqrt(6)

    @property
    def look_at(self):
        return self._look_at

    @look_at.setter
    def look_at(self, value: np.ndarray):
        self._look_at = value

    @property
    def projection(self):
        return self._projection

    @projection.setter
    def projection(self, value: np.ndarray):
        self._projection = value

    @property
    def scene(self):
        return self._scene

    @scene.setter
    def scene(self, value: tuple[Mesh] | list[Mesh]):
        self._scene = value

    @property
    def viewport(self):
        return self._viewport

    @viewport.setter
    def viewport(self, value):
        self._viewport = value

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
        return cls(
            look_at=cls.ISOMETRIC_VIEW_MATRIX,
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


class Engine:
    def __init__(self, views, precision=7):
        self._views = views
        self._precision = precision

    @property
    def views(self):
        return self._views

    @views.setter
    def views(self, views):
        self._views = views

    @property
    def precision(self):
        return self._precision

    def render(self, filename, size=(512, 512), viewbox="-0.5 -0.5 1.0 1.0", **extra):
        drawing = svgwrite.Drawing(filename, size, viewBox=viewbox, **extra)
        self._draw(drawing)
        drawing.save()
        print(f"Wrote file {filename}")

    def _draw(self, drawing):
        for view in self.views:
            projection = np.dot(view.look_at, view.projection)
            # Initialize clip path. See https://www.w3.org/TR/SVG11/masking.html#ClippingPaths
            clip_path = drawing.defs.add(drawing.clipPath())
            clip_min = view.viewport.minx, view.viewport.miny
            clip_size = view.viewport.width, view.viewport.height
            clip_path.add(drawing.rect(clip_min, clip_size))

            for mesh in view.scene:
                group = self._create_group(drawing, projection, view.viewport, mesh)
                group["clip-path"] = clip_path.get_funciri()
                drawing.add(group)

    def _create_group(self, drawing, projection, viewport, mesh):
        faces = mesh.faces
        shader = mesh.shader or (lambda face_index, mesh: {})
        default_style = mesh.style or {}

        # Extend each point to a vec4, then transform to clip space.
        faces = np.dstack([faces, np.ones(faces.shape[:2])])
        faces = np.dot(faces, projection)

        # Reject trivially clipped polygons.
        xyz, w = faces[:, :, :3], faces[:, :, 3:]
        accepted = (xyz > -w) & (xyz < +w)
        accepted = accepted.all(axis=2)  # vert is accepted if xyz are all inside
        accepted = accepted.any(axis=1)  # face is accepted if any vert is inside
        degenerate = (w <= 0)[:, :, 0]  # vert is bad if its w <= 0
        degenerate = degenerate.any(axis=1)  # face is bad if any of its verts are bad
        accepted = np.logical_and(accepted, np.logical_not(degenerate))

        faces = np.compress(accepted, faces, axis=0)
        if len(faces) == 0:
            warnings.warn(
                "All faces were pruned! Check your projection matrix.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Apply perspective transformation.
        xyz, w = faces[:, :, :3], faces[:, :, 3:]
        faces = xyz / w

        # Sort faces from back to front.
        face_indices = self._sort_back_to_front(faces)
        faces = faces[face_indices]

        # Apply viewport transform to X and Y.
        faces[:, :, 0:1] = (1.0 + faces[:, :, 0:1]) * viewport.width / 2
        faces[:, :, 1:2] = (1.0 - faces[:, :, 1:2]) * viewport.height / 2
        faces[:, :, 0:1] += viewport.minx
        faces[:, :, 1:2] += viewport.miny

        # Compute the winding direction of each polygon.
        # windings = np.zeros(faces.shape[0])
        # if faces.shape[1] >= 3:
        #     normals = np.compress(accepted, mesh.normals, axis=0)
        #     np.copyto(windings, normals[:, 2])

        group = drawing.g(**default_style)

        # Create circles.
        if mesh.circle_radius > 0:
            for face_index, face in enumerate(faces):
                style = shader(face_indices[face_index], mesh)
                if style is None:
                    continue
                face = np.around(face[:, :2], self.precision)
                for pt in face:
                    group.add(drawing.circle(pt, mesh.circle_radius, **style))
            return group

        # Create polygons and lines.
        for face_index, face in enumerate(faces):
            style = shader(face_indices[face_index], mesh)
            if style is None:
                continue
            face = np.around(face[:, :2], self.precision)
            _, indices = np.unique(face, return_index=True, axis=0)
            face = face[sorted(indices)]
            if len(face) == 2:
                group.add(drawing.line(face[0], face[1], **style))
            else:
                group.add(drawing.polygon(face, **style))
        return group

    def _sort_back_to_front(self, faces):
        z_centroids = -np.sum(faces[:, :, 2], axis=1)
        for face_index in range(len(z_centroids)):
            z_centroids[face_index] /= len(faces[face_index])
        return np.argsort(z_centroids)


_directional_light = np.array([2, 2, 1]) / 2


def _hex2rgb(hexc):
    hexc = hexc.lstrip("#")
    return np.array([int(hexc[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0


def _rgb2hex(rgb):
    rgb = (rgb * 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*rgb).upper()


def _apply_shading(base_color, shading, factor=0.5):
    # `shading` is a value between -1 and 1
    # factor controls how much lighter/darker we go from the base color
    base_rgb = _hex2rgb(base_color)
    shaded_color = base_rgb + factor * shading * (np.ones(3) - base_rgb)

    shaded_color = np.clip(shaded_color, 0, 1)  # Ensure RGB values are within [0, 1]
    return _rgb2hex(shaded_color)


BASE_COLOR = "#71618D"
base_style = {}


def shader(face_index, mesh, base_color="#71618D"):
    mesh = mesh.faces

    # TODO
    normal = mesh.normals[face_index] / np.linalg.norm(mesh.normals[face_index])
    shading = np.dot(normal, _directional_light)

    new_color = _apply_shading(base_color, shading, factor=0.6)

    return base_style | {"fill": new_color}


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
