from coxeter.shapes import ConvexPolyhedron
import numpy as np

import svg3d
style = {
    "fill": "#00B2A6",
    "fill_opacity": "0.95",
    "stroke": "black",
    "stroke_linejoin": "round",
    "stroke_width": "0.005",
}

flat_shader = lambda face_index, mesh: style

# Create 12 cubes spaced by 0.5 units along the x-axis
# Cube side length is 1, so spacing of 0.5 means gap of 0.5 between cubes
# Position each cube at x = i * (1 + 0.5) = i * 1.5
scene = []
SPACE = 2.5
N = 11


from conway.seeds.dodecahedra import make_vertices_tetartoid_family as tetar


ssps = np.linspace(-1.5, 1.0, N)[::-1]

for i in range(N):
    offset = np.array([i * SPACE - ((N-1) * SPACE / 2) , 0.0, 0.0])
    poly = ConvexPolyhedron(tetar(ssps[i]))
    poly.minimal_centered_bounding_sphere_radius = 1.0
    poly.centroid = offset
    cube_mesh = svg3d.Mesh.from_coxeter(
        poly, shader=flat_shader
    )
    scene.append(cube_mesh)

view = svg3d.View.orthographic(
    scene=scene,
    scene_width=32,
    aspect_ratio=4.0,  # 32/16
    theta=90,  # rotation in xy plane
    elevation=20,  # angle from horizontal (0=side, 90=top-down)
)

# Use viewport's recommended size to match the aspect ratio
svg3d.Engine([view]).render("line_of_cubes.svg", size=view.viewport.get_size())
print("Rendered line_of_cubes.svg")
