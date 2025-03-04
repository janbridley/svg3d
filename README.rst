.. SVG3D

.. container:: row

   .. image:: https://raw.githubusercontent.com/janbridley/svg3d/refs/heads/main/doc/source/_static/cube-wireframe.svg
      :alt: Cube Wireframe
      :width: 17%

   .. image:: https://raw.githubusercontent.com/janbridley/svg3d/refs/heads/main/doc/source/_static/cycle-compact.svg
      :alt: Alternation Cycle
      :width: 17%

   .. image:: https://raw.githubusercontent.com/janbridley/svg3d/refs/heads/main/doc/source/_static/CrumpledDevelopable-tri-compact.svg
      :alt: Keenan CrumpledDevelopable
      :width: 17%

   .. image:: https://raw.githubusercontent.com/janbridley/svg3d/refs/heads/main/doc/source/_static/oloid_64-tri-compact.svg
      :alt: Keenan Oloid
      :width: 17%

   .. image:: https://raw.githubusercontent.com/janbridley/svg3d/refs/heads/main/doc/source/_static/bunny-tri-compact.svg
      :alt: Stanford Bunny
      :width: 17%

.. _imheader:

SVG3D was designed to bridge the gap between raytraced rendering engines like Blender and plotting tools like matplotlib and plotly. Common computer graphics techniques and models have been adapted to work within the counstraints of vector art, an approach that enables users to generate compact, scalable images with realistic shading.

A reimagining of the excellent `original library <https://prideout.net/blog/svg_wireframes/#using-the-api>`_ with the same name, this version has many new features, a more general interface, and a somewhat different scope. We aim to streamline the process of rendering scenes of geometries for scientific publications, although the libary is useful for a diverse array of applications.

Many thanks to the `Keenan 3D Model repository <https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/>`_ and the `Georgia Tech Large Models Archive <https://sites.cc.gatech.edu/projects/large_models/>`_ for the models rendered in the header image.

|CI|
|ReadTheDocs|

.. |CI| image:: https://github.com/janbridley/svg3d/actions/workflows/run-pytest.yaml/badge.svg
   :target: https://github.com/janbridley/svg3d/actions
.. |ReadTheDocs| image:: https://readthedocs.org/projects/svg3d/badge/?version=latest
   :target: http://svg3d.readthedocs.io/en/latest/?badge=latest

.. _installing:

Installation
============

`svg3d` is available on PyPI, and can be easily installed from there:

.. code-block:: bash

   pip install svg3d


The package can also be built from source:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/janbridley/svg3d.git
   cd svg3d

   # Install to your python environment!
   python -m pip install .

.. _quickstartexample:

Quickstart Example
==================

`svg3d` provides convenience `View` options for standard rendering perspectives - isometric, dimetric, and trimetric. Shapes can be easily created from coxeter objects, or from raw mesh data.

.. code-block:: python

   from coxeter.families import ArchimedeanFamily
   import svg3d

   style = {
       "fill": "#00B2A6",
       "fill_opacity": "0.85",
       "stroke": "black",
       "stroke_linejoin": "round",
       "stroke_width": "0.005",
   }

   truncated_cube = ArchimedeanFamily.get_shape("Truncated Cube")

   scene = [
       svg3d.Mesh.from_coxeter(
           truncated_cube, style=style, shader=svg3d.shaders.diffuse_lighting
       )
   ]

   # Convenience views: isometric, dimetric, and trimetric
   iso = svg3d.View.isometric(scene, fov=1.0)
   dim = svg3d.View.dimetric(scene, fov=1.0)
   tri = svg3d.View.trimetric(scene, fov=1.0)

   for view, view_type in zip([iso, dim, tri], ["iso", "dim", "tri"]):
       svg3d.Engine([view]).render(f"{view_type}.svg")

.. list-table::
   :header-rows: 1

   * - Isometric
     - Dimetric
     - Trimetric
   * - .. image:: https://raw.githubusercontent.com/janbridley/svg3d/refs/heads/main/doc/source/_static/iso.svg
     - .. image:: https://raw.githubusercontent.com/janbridley/svg3d/refs/heads/main/doc/source/_static/dim.svg
     - .. image:: https://raw.githubusercontent.com/janbridley/svg3d/refs/heads/main/doc/source/_static/tri.svg

.. _usageexample:

Usage Example
=============

In addition to convenience methods, `svg3d` allows full control over the viewport, scene geometry, image style, and shaders. Methods are based on OpenGL standards and nomenclature where possible, and images can be created from any set of vertices and faces - even from ragged arrays! Simply pass an array of vertices and a list of arrays (one for vertex indices of each face, as below) to `svg3d.Mesh.from_vertices_and_faces` to render whatever geometry you like. Custom shader models can be implemented as a callable that takes a face index and a `svg3d.Mesh` object to shade.

.. code-block:: python

   import numpy as np
   import svg3d

   # Define the vertices and faces of a cube
   vertices = np.array(
       [[-1., -1., -1.],
       [-1., -1.,  1.],
       [-1.,  1., -1.],
       [-1.,  1.,  1.],
       [ 1., -1., -1.],
       [ 1., -1.,  1.],
       [ 1.,  1., -1.],
       [ 1.,  1.,  1.]]
   )

   faces = [
       [0, 2, 6, 4],
       [0, 4, 5, 1],
       [4, 6, 7, 5],
       [0, 1, 3, 2],
       [2, 3, 7, 6],
       [1, 5, 7, 3]
   ]

   # Set up our rendering style - transparent white gives a nice wireframe appearance
   style = {
       "fill": "#FFFFFF",
       "fill_opacity": "0.75",
       "stroke": "black",
       "stroke_linejoin": "round",
       "stroke_width": "0.005",
   }

   empty_shader = lambda face_index, mesh: {} # Does nothing, but illustrates the shader API

   pos_object = [0.0, 0.0, 0.0]  # "at" position
   pos_camera = [40, 40, 120]  # "eye" position
   vec_up = [0.0, 1.0, 0.0]  # "up" vector of camera. This is the default value.

   z_near, z_far = 1.0, 200.0
   aspect = 1.0  # Aspect ratio of the view cone
   fov_y = 2.0  # Opening angle of the view cone. fov_x is equal to fov_y * aspect

   look_at = svg3d.get_lookat_matrix(pos_object, pos_camera, vec_up=vec_up)
   projection = svg3d.get_projection_matrix(
       z_near=z_near, z_far=z_far, fov_y=fov_y, aspect=aspect
   )

   # A "scene" is a list of Mesh objects, which can be easily generated from raw data
   scene = [
       svg3d.Mesh.from_vertices_and_faces(vertices, faces, style=style, shader=empty_shader)
   ]

   view = svg3d.View.from_look_at_and_projection(
       look_at=look_at,
       projection=projection,
       scene=scene,
   )

   svg3d.Engine([view]).render("cube-wireframe.svg")

Running the code above generates the following image:

.. image:: https://raw.githubusercontent.com/janbridley/svg3d/refs/heads/main/doc/source/_static/cube-wireframe.svg
