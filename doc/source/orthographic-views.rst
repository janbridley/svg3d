.. _orthographic_views:

Orthographic Views
==================

Orthographic (parallel) projection is the foundation of technical and scientific visualization.
Unlike perspective projection, orthographic views preserve measurements and parallel lines,
making them ideal for diagrams, engineering drawings, and scientific publications.

This guide covers the ``View.orthographic`` API and the convenience methods for standard
axonometric projections.

.. _orthographic_basics:

Basic Orthographic View
-----------------------

The ``View.orthographic`` method provides full control over camera positioning using
intuitive cylindrical coordinates:

.. code-block:: python

   import svg3d

   view = svg3d.View.orthographic(
       scene=scene,
       scene_width=3.0,       # Width of view volume in world units
       aspect_ratio=1.0,      # Width/height ratio of viewport
       theta=45.0,            # Azimuthal angle (rotation in xy plane)
       elevation=35.264,      # Angle from xy plane toward +z
   )

   svg3d.Engine([view]).render("output.svg")

**Parameters:**

- ``scene``: List of Mesh objects to render
- ``scene_width``: Width of the view volume (larger = more zoomed out)
- ``aspect_ratio``: Width/height ratio of the viewport
- ``theta``: Azimuthal angle in degrees (rotation in xy plane)
- ``elevation``: Angle from xy plane toward +z in degrees

.. _theta_elevation:

Understanding Theta and Elevation
---------------------------------

The camera position is controlled by two angles:

**Theta (θ)** - Azimuthal Rotation
  Controls rotation around the z-axis (in the xy plane):

  - ``theta=0°``: Camera looks from +x direction
  - ``theta=90°``: Camera looks from +y direction
  - ``theta=180°``: Camera looks from -x direction
  - ``theta=270°``: Camera looks from -y direction

**Elevation (φ)** - Vertical Angle
  Controls the angle from the xy plane toward +z:

  - ``elevation=0°``: Horizontal view (side view)
  - ``elevation=45°``: Mid-angle view
  - ``elevation=90°``: Top-down view (looking straight down +z)

.. list-table::
   :header-rows: 1

   * - Elevation
     - Effect
   * - 0°
     - Side view: z-axis appears vertical
   * - 35.264°
     - Isometric: all axes equally foreshortened
   * - 90°
     - Top-down: z-axis collapses to a point

**Example: Varying Theta**

.. code-block:: python

   # Four views at 90° intervals around the scene
   for theta in [0, 90, 180, 270]:
       view = svg3d.View.orthographic(
           scene=scene,
           scene_width=3.0,
           aspect_ratio=1.0,
           theta=theta,
           elevation=30.0,
       )
       svg3d.Engine([view]).render(f"view_theta_{theta}.svg")

.. _axonometric:

Standard Axonometric Projections
--------------------------------

svg3d provides convenience methods for the three standard axonometric projections,
all using true orthographic (parallel) projection:

Isometric Projection
^^^^^^^^^^^^^^^^^^^^

All three coordinate axes are equally foreshortened. This is achieved with
``theta=45°`` and ``elevation=35.264°`` (arcsin(1/√3)).

.. code-block:: python

   # Isometric: all axes equally foreshortened
   view = svg3d.View.isometric(scene, scene_width=4.0)

.. list-table::
   :header-rows: 1

   * - Isometric View
   * - .. image:: _static/iso.svg

Dimetric Projection
^^^^^^^^^^^^^^^^^^^

Two coordinate axes are equally foreshortened, the third has different foreshortening.
Uses ``theta=45°`` and ``elevation=20.705°`` (arcsin(1/√8)).

.. code-block:: python

   # Dimetric: two axes equally foreshortened
   view = svg3d.View.dimetric(scene, scene_width=4.0)

.. list-table::
   :header-rows: 1

   * - Dimetric View
   * - .. image:: _static/dim.svg

Trimetric Projection
^^^^^^^^^^^^^^^^^^^^

All three coordinate axes have different foreshortening. Uses ``theta=30°`` and
``elevation=25°`` for a natural-looking view.

.. code-block:: python

   # Trimetric: all axes differently foreshortened
   view = svg3d.View.trimetric(scene, scene_width=4.0)

.. list-table::
   :header-rows: 1

   * - Trimetric View
   * - .. image:: _static/tri.svg

.. _scene_width_aspect:

Scene Width and Aspect Ratio
-----------------------------

Scene Width
^^^^^^^^^^^

The ``scene_width`` parameter controls the horizontal extent of the view volume.
Larger values show more of the scene (zoom out), smaller values show less (zoom in).

.. code-block:: python

   # Zoomed in (small scene_width)
   view_close = svg3d.View.orthographic(
       scene=scene,
       scene_width=2.0,  # Close view
       theta=45.0,
       elevation=35.264,
   )

   # Zoomed out (large scene_width)
   view_far = svg3d.View.orthographic(
       scene=scene,
       scene_width=8.0,  # Wide view
       theta=45.0,
       elevation=35.264,
   )

Aspect Ratio
^^^^^^^^^^^^

The ``aspect_ratio`` parameter controls the width/height ratio of the viewport.
Use this to match your output image dimensions.

.. code-block:: python

   # Square viewport
   view_square = svg3d.View.orthographic(
       scene=scene,
       scene_width=4.0,
       aspect_ratio=1.0,  # Square
       theta=45.0,
       elevation=35.264,
   )

   # Wide viewport (e.g., for panoramic views)
   view_wide = svg3d.View.orthographic(
       scene=scene,
       scene_width=4.0,
       aspect_ratio=2.0,  # 2:1 width:height
       theta=45.0,
       elevation=35.264,
   )

   # Render with matching pixel dimensions
   svg3d.Engine([view_wide]).render(
       "wide_view.svg",
       size=view_wide.viewport.get_size(base_height=512)
   )

.. _custom_views:

Custom Views
------------

You can create custom axonometric views by choosing specific theta and elevation
values tailored to your visualization needs.

Emphasizing Specific Faces
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Emphasize top face (high elevation)
   view_top = svg3d.View.orthographic(
       scene=scene,
       scene_width=3.0,
       aspect_ratio=1.0,
       theta=45.0,
       elevation=70.0,  # High elevation shows more of top
   )

   # Emphasize front face (low elevation)
   view_front = svg3d.View.orthographic(
       scene=scene,
       scene_width=3.0,
       aspect_ratio=1.0,
       theta=45.0,
       elevation=15.0,  # Low elevation shows more of front
   )

Game-Style Isometric
^^^^^^^^^^^^^^^^^^^^

Classic isometric games often use approximately 30° elevation for a pleasing look:

.. code-block:: python

   # Classic game-style isometric view
   view_game = svg3d.View.orthographic(
       scene=scene,
       scene_width=3.0,
       aspect_ratio=1.0,
       theta=45.0,
       elevation=30.0,
   )

.. _common_configurations:

Common Configurations Reference
-------------------------------

+------------------+--------+------------+-------------------------------------+
| View Type        | theta  | elevation  | Description                         |
+==================+========+============+=====================================+
| Isometric        | 45°    | 35.264°    | All axes equally foreshortened      |
+------------------+--------+------------+-------------------------------------+
| Dimetric         | 45°    | 20.705°    | Two axes equally foreshortened      |
+------------------+--------+------------+-------------------------------------+
| Trimetric        | 30°    | 25°        | All axes differently foreshortened  |
+------------------+--------+------------+-------------------------------------+
| Top-down         | any    | 90°        | Looking straight down (+z)          |
+------------------+--------+------------+-------------------------------------+
| Side view        | any    | 0°         | Horizontal view from xy plane       |
+------------------+--------+------------+-------------------------------------+
| Front view       | 0°     | 0°         | Looking from +x toward origin       |
+------------------+--------+------------+-------------------------------------+
| Right view       | 90°    | 0°         | Looking from +y toward origin       |
+------------------+--------+------------+-------------------------------------+

.. _complete_example:

Complete Example
----------------

Here's a complete example that generates multiple views of a truncated cube:

.. code-block:: python

   from coxeter.families import ArchimedeanFamily
   import svg3d

   # Create the shape
   truncated_cube = ArchimedeanFamily.get_shape("Truncated Cube")

   style = {
       "fill": "#00B2A6",
       "fill_opacity": "0.85",
       "stroke": "black",
       "stroke_linejoin": "round",
       "stroke_width": "0.005",
   }

   scene = [
       svg3d.Mesh.from_coxeter(
           truncated_cube,
           shader=svg3d.shaders.DiffuseShader.from_style_dict(style)
       )
   ]

   # Generate standard axonometric views
   for method_name, method in [
       ("isometric", svg3d.View.isometric),
       ("dimetric", svg3d.View.dimetric),
       ("trimetric", svg3d.View.trimetric),
   ]:
       view = method(scene, scene_width=3.0)
       svg3d.Engine([view]).render(f"{method_name}.svg")

   # Generate a custom view
   custom_view = svg3d.View.orthographic(
       scene=scene,
       scene_width=3.0,
       aspect_ratio=1.0,
       theta=60.0,
       elevation=40.0,
   )
   svg3d.Engine([custom_view]).render("custom_view.svg")
