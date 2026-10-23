"""Regression tests for :obj:`~.Mesh` face ingestion (BUGS.md A4, A5, A6).

The mesh must accept any face input that its own API declares legal: ragged
lists of arrays through both the constructor and the ``faces`` setter (A4),
meshes made purely of 2-vertex line segments (A5), and empty face lists (A6).
"""

import warnings

import numpy as np
import numpy.testing as npt

from svg3d import Engine, Mesh, View

TRIANGLE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
QUAD = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
SEGMENT = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
SEGMENTS = [
    SEGMENT,
    np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
]


def render_scene(meshes, tmp_path, name):
    """Render ``meshes`` isometrically and return the SVG text."""
    view = View.isometric(meshes)
    filename = tmp_path / name
    with warnings.catch_warnings():
        # Any warning (notably "All faces were pruned!") is a failure here.
        warnings.simplefilter("error")
        Engine([view]).render(str(filename))
    return filename.read_text()


def test_faces_setter_accepts_ragged_list_of_arrays():
    mesh = Mesh(faces=[TRIANGLE])
    mesh.faces = [TRIANGLE, QUAD]
    reference = Mesh(faces=[TRIANGLE, QUAD])
    assert mesh.faces.shape == (2, 4, 3)
    npt.assert_array_equal(mesh.faces, reference.faces)
    npt.assert_array_equal(mesh.normals, reference.normals)


def test_faces_setter_getter_matches():
    mesh = Mesh(faces=[TRIANGLE, QUAD])
    padded = mesh.faces
    mesh.faces = padded
    npt.assert_array_equal(mesh.faces, padded)
    assert mesh.normals.shape == (2, 3)


def test_pure_wireframe_mesh_has_zero_normals():
    mesh = Mesh(faces=SEGMENTS)
    assert mesh.faces.shape == (2, 2, 3)
    assert mesh.normals.shape == (2, 3)
    npt.assert_array_equal(mesh.normals, 0.0)


def test_pure_wireframe_mesh_renders_as_lines(tmp_path):
    svg = render_scene([Mesh(faces=SEGMENTS)], tmp_path, "wireframe.svg")
    assert svg.count("<line") == 2
    assert "<polygon" not in svg


def test_empty_mesh_is_constructible():
    mesh = Mesh(faces=[])
    assert mesh.faces.shape == (0, 0, 3)
    assert mesh.normals.shape == (0, 3)


def test_empty_face_list_is_assignable():
    mesh = Mesh(faces=[TRIANGLE])
    mesh.faces = []
    assert mesh.faces.shape == (0, 0, 3)
    assert mesh.normals.shape == (0, 3)


def test_empty_mesh_renders_blank_svg_without_warning(tmp_path):
    svg = render_scene([Mesh(faces=[])], tmp_path, "blank.svg")
    assert "<polygon" not in svg
    assert "<line" not in svg


def test_mixed_degree_mesh_behavior_is_unchanged():
    mesh = Mesh(faces=[SEGMENT, TRIANGLE])
    # The line segment is still padded out to the triangle's vertex count
    assert mesh.faces.shape == (2, 3, 3)
    # and still gets a zero normal, as does a pure wireframe face.
    npt.assert_array_equal(mesh.normals[0], 0.0)
    npt.assert_array_equal(mesh.normals[1], [0.0, 0.0, 1.0])
