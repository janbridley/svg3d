import subprocess

import svg3d

# The style of our SVG images can be stored in a dictionary object
style = {
    "fill": "#71618D",
    "fill_opacity": "0.85",
    "stroke": "black",
    "stroke_linejoin": "round",
    "stroke_width": "0.005",
}

# cubo = ArchimedeanFamily.get_shape("Cuboctahedron")

# print(cubo.vertices)

mesh = svg3d.Mesh.example_mesh(triangulated=True)
mesh.shader = svg3d.shaders.DiffuseShader.from_style_dict(style)

view = svg3d.View.dimetric(
    scene=[mesh],
)

filename = "test.svg"
svg3d.Engine([view]).render(filename)

subprocess.run(["open", filename])
