This is a single-file Python library for generating 3D wireframes in SVG format. We aim to create a framework for rendering shapes (and arrays of shapes) as vector graphics in a performant and pythonic way. This code is adapted from [the original svg3d](https://github.com/prideout/svg3d) for compatibility with [coxeter](https://github.com/glotzerlab/coxeter) and to simplify the interface.


For a description of how the library was initially designed and implemented, check out
[this blog post](https://prideout.net/blog/svg_wireframes/#using-the-api).

## Usage example (ConvexPolyhedron)
```python
from coxeter.families import ArchimedeanFamily

import svg3d
from svg3d import get_lookat_matrix, get_projection_matrix


def generate_svg(filename, poly):
    pos_object = [0.0, 0.0, 0.0]  # "at" position
    pos_camera = [40, 40, 120]  # "eye" position
    vec_up = [0.0, 1.0, 0.0]  # "up" vector of camera. This is the default value.

    z_near, z_far = 1.0, 200.0
    aspect = 1.0  # Aspect ratio of the view cone
    fov_y = 1.0  # Opening angle of the view cone. fov_x is equal to fov_y * aspect

    look_at = get_lookat_matrix(pos_object, pos_camera, vec_up=vec_up)
    projection = get_projection_matrix(
        z_near=z_near, z_far=z_far, fov_y=fov_y, aspect=aspect
    )

    # A "scene" is a list of Mesh objects, which can be easily generated from Coxeter!
    scene = [svg3d.Mesh.from_poly(poly, style=style)]

    view = svg3d.View.from_look_at_and_projection(
        look_at=look_at,
        projection=projection,
        scene=scene,
    )

    svg3d.Engine([view]).render(filename)


style = dict(
    fill="#00B2A6",
    fill_opacity="0.85",
    stroke="black",
    stroke_linejoin="round",
    stroke_width="0.005",
)

truncated_cube = ArchimedeanFamily.get_shape("Truncated Cube")
generate_svg(filename="truncated_cube.svg", poly=truncated_cube)
```

And that's all it takes! For an even simpler startup, use one of the built in viewports.

For example:

```python
import svg3d

scene = [svg3d.Mesh.from_poly(poly, style=style)]

iso = svg3d.View.isometric(scene, fov=1.0)
dim = svg3d.View.dimetric(scene, fov=1.0)
tri = svg3d.View.trimetric(scene, fov=1.0)

```
