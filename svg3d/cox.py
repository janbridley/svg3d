from coxeter.shapes import ConvexPolyhedron
from .svg3d import Camera, Mesh, View, Engine, Scene
import numpy as np
import pyrr


def pad_poly_faces(poly):
    padl = max([len(face) for face in poly.faces])
    padded_faces = []
    for face in poly.faces:
        if len(face) == padl:
            padded_faces.append(face)
        face = list(face)

        while len(face) < padl:
            face.append(face[0])
        padded_faces.append(face)
    padded_faces = np.array(padded_faces)
    return padded_faces


def generate_svg_from_coxeter(
    poly: ConvexPolyhedron, filename: str, style: dict = None
):
    view = pyrr.matrix44.create_look_at(
        # eye=[50, 40, 120], target=[0, 0, 0], up=[0, 1, 0]
        eye=[50, 40, 120],
        target=[0, 0, 0],
        up=[0, 1, 0],
    )
    projection = pyrr.matrix44.create_perspective_projection(
        fovy=2, aspect=1, near=10, far=200
    )
    camera = Camera(view, projection)

    if style is None:  # use defaults
        style = dict(
            fill="#FFFFFF",
            fill_opacity="0.85",
            stroke="black",
            stroke_linejoin="round",
            stroke_width="0.005",
        )

    vertices = poly.vertices
    padded_faces = pad_poly_faces(poly)
    faceverts = vertices[padded_faces]

    mesh = Mesh(faceverts, style=style)
    view = View(camera, Scene([mesh]))
    Engine([view]).render(filename)
