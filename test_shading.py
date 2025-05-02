import subprocess

from coxeter.families import CatalanFamily
from svgwrite.gradients import LinearGradient

import svg3d

# The style of our SVG images can be stored in a dictionary object
style = {
    "fill": "#494452",#"#71618D",
    "fill_opacity": "1.0",
    "stroke": "black",
    "stroke_linejoin": "round",
    "stroke_width": "0.005",
}

# cubo = ArchimedeanFamily.get_shape("Cuboctahedron")
# kico = CatalanFamily.get_shape("Triakis Icosahedron")
poly = CatalanFamily.get_shape("Pentakis Dodecahedron")

# print(cubo.vertices)

# mesh = svg3d.Mesh.example_mesh(triangulated=True)
mesh = svg3d.Mesh.from_coxeter(poly)
mesh.shader = svg3d.shaders.DiffuseShader.from_style_dict(style)
mesh.shader.absorbance = 0.6

view = svg3d.View.dimetric(
    scene=[mesh],
)

filename = "test.svg"
svg3d.Engine([view]).render(filename)

subprocess.run(["open", filename]) # noqa: S603, S607

# So it seems like the gradients are ALWAYS one-dimensional
# So - what options do we have for creating the gradient.
# 1. gradientTransform with a rotation to move along face
# 2. x1, x2, y1, y2: 2d coordinates for start and end of gradient ()

"""
 'Instead of computing illumination only at the center of a triangle, we can
compute illumination at its three vertices. This gives us three illumination
values between 0.0 and 1.0, one for each vertex of the triangle.'

As here: https://www.alecjacobson.com/weblog/3398.html

It seems like we just define a gradient that travels along each edge?

Or as here https://stackoverflow.com/a/3183535/21897583,
it is possible the gradient should travel from the vertex to the center of the opposite
edge
"""

# grad = LinearGradient(start=())
