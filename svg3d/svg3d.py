# Adapted from https://prideout.net/blog/svg_wireframes/
# Copyright (c) 2019 Philip Rideout. Modified 2024 by Jenna Bradley.
# Distributed under the MIT License, see bottom of file.

import warnings
from typing import TYPE_CHECKING, Callable

import numpy as np
import svgwrite
import tqdm

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


class Mesh:
    def __init__(
        self,
        faces: list[np.ndarray],
        shader: Callable[[int, float], dict] | None = None,
        style: dict | None = None,
        circle_radius: float = 0.0,
    ):
        self._faces = _pad_arrays(faces)
        self._compute_normals()
        self._shader = shader
        self._style = style
        self._circle_radius = circle_radius

    @property
    def faces(self):
        """np.ndarray: Get or set the faces of the :obj:`~.Mesh`"""
        return self._faces

    @faces.setter
    def faces(self, faces):
        self._faces = faces
        self._compute_normals()

    @property
    def shader(self):
        """:py:obj:`~typing.Callable`: Get or set the :obj:`~.Shader` for the \
        :obj:`~.Mesh`"""
        return self._shader

    @shader.setter
    def shader(self, shader):
        self._shader = shader

    @property
    def style(self):
        """dict: Get or set the style dictionary for the mesh."""
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
        """np.ndarray: Get the normals for the faces of the :obj:`~.Mesh`."""
        return self._normals

    def _compute_normals(self):
        face_simplices = self.faces[:, :3]

        # Convert each simplex (3 points) into two edge vectors (each 2 points)
        # These will be an array of [N, (0-1,1-2)=2, 3] vertices
        face_edge_vectors = np.diff(face_simplices, axis=1)

        # The LSP is unhappy, but this is correct. Each face has exactly 2 edge vectors
        normals = np.cross(*np.split(face_edge_vectors, 2, axis=1)).squeeze()

        self._normals = normals / np.linalg.norm(normals)  # Return normalized

    @classmethod
    def from_coxeter(
        cls,
        poly: "coxeter.shapes.ConvexPolyhedron",
        shader: Callable[[int, float], dict] | None = None,
        style: dict | None =None,
    ):
        """Create a :obj:`~.Mesh` object from a coxeter
        :class:`~coxeter.shapes.ConvexPolyhedron`."""
        return cls(
            faces=[poly.vertices[face] for face in poly.faces],
            shader=shader,
            style=style,
        )

    @classmethod
    def from_vertices_and_faces(
        cls,
        vertices: np.ndarray[float],
        faces: list[np.ndarray[int]],
        shader: Callable[[int, float], dict] | None = None,
        style: dict | None =None,
    ):
        return cls(
            faces=[vertices[face] for face in faces],
            shader=shader,
            style=style,
        )


class Engine:
    def __init__(self, views, precision=10):
        self._views = views
        self._precision = precision

    @property
    def views(self):
        """list[:obj:`~.View`]: Get or set the list of views to render."""
        return self._views

    @views.setter
    def views(self, views):
        self._views = views

    @property
    def precision(self):
        """int: Get or set the rounding precision for vertices of rendered polygons."""
        return self._precision

    @precision.setter
    def precision(self, precision):
        return self._precision

    def render(self, filename, size=(512, 512), viewbox="-0.5 -0.5 1.0 1.0", **extra):
        """
        Render the current view or views to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the render to. Should be postfixed with `.svg`
        size : tuple of int, optional
            Size of the render in pixels. Default is (512, 512).
        viewbox : str, optional
            :class:`~svgwrite.mixins.viewBox` attribute for the SVG. Default is \
            "-0.5 -0.5 1.0 1.0".
        **extra
            Additional keyword arguments to be passed into :py:mod:`svgwrite`.

        Raises
        ------
        RuntimeWarning
            If all faces are pruned due to incorrect projection matrix.
        """
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
        for face_index, face in enumerate(tqdm.tqdm(faces)):
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
