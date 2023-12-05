<img src="https://prideout.net/blog/svg_wireframes/filmstrip.svg" width="512px">

This is a ~~single-file Python library~~ _pip-installable Python package_ for generating 3D wireframes in SVG format. _We aim to create a framwork for rendering shapes (and arrays of shapes) as vector graphics in a performant and pythonic way._

For a description of how the library was designed and implemented, check out
[this blog post](https://prideout.net/blog/svg_wireframes/#using-the-api).

## Usage example

```python
import numpy, svg3d, pyrr, math

def get_octahedron_faces():
    f = math.sqrt(2.0) / 2.0
    verts = numpy.float32([ ( 0, -1,  0), (-f,  0,  f), ( f,  0,  f), ( f,  0, -f), (-f,  0, -f), ( 0,  1,  0) ])
    triangles = numpy.int32([ (0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 1, 4), (5, 1, 2), (5, 2, 3), (5, 3, 4), (5, 4, 1) ])
    return 15.0 * verts[triangles]

def generate_svg(filename):
    view = pyrr.matrix44.create_look_at(eye=[50, 40, 120], target=[0, 0, 0], up=[0, 1, 0])
    projection = pyrr.matrix44.create_perspective_projection(fovy=15, aspect=1, near=10, far=200)
    camera = svg3d.Camera(view, projection)

    style = dict(
        fill="white", fill_opacity="0.75",
        stroke="black", stroke_linejoin="round", stroke_width="0.005")

    mesh = svg3d.Mesh(get_octahedron_faces(), style=style)
    view = svg3d.View(camera, svg3d.Scene([mesh]))
    svg3d.Engine([view]).render(filename)

generate_svg("octahedron.svg")
```

The above code snippet generates an image like this:

<img src="https://prideout.net/blog/svg_wireframes/octahedron.svg" width="256px">

## Running the test script

```
pipenv shell
pipenv install
cd extras
./test.py && open gallery.html
```
