#!/usr/bin/env python3
"""
Generate example images for the orthographic views documentation.

This script creates SVG files demonstrating theta rotation, elevation changes,
and comparison views for the ReadTheDocs documentation.
"""

import numpy as np
from coxeter.families import ArchimedeanFamily

import svg3d

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = "source/_static"

STYLE = {
    "fill": "#00B2A6",
    "fill_opacity": "0.85",
    "stroke": "black",
    "stroke_linejoin": "round",
    "stroke_width": "0.005",
}


def create_scene():
    """Create a scene with a truncated cube."""
    shape = ArchimedeanFamily.get_shape("Truncated Cube")
    return [
        svg3d.Mesh.from_coxeter(
            shape,
            shader=svg3d.shaders.DiffuseShader.from_style_dict(STYLE)
        )
    ]


def render_grid(views, filename, positions, size=(1024, 512)):
    """Render multiple views in a grid layout."""
    for view, vp in zip(views, positions):
        view.viewport = vp
    svg3d.Engine(views).render(filename, size=size)
    print(f"Wrote {filename}")


# =============================================================================
# Generate Images
# =============================================================================

def generate_theta_rotation():
    """Generate 4 views showing theta rotation at 90° intervals."""
    scene = create_scene()

    theta_values = [0, 90, 180, 270]
    views = []

    for theta in theta_values:
        view = svg3d.View.orthographic(
            scene=scene,
            scene_width=3.0,
            aspect_ratio=1.0,
            theta=theta,
            elevation=30.0,
        )
        views.append(view)

    # Side by side (1x4)
    positions = [
        svg3d.view.Viewport(-2.0, -0.5, 0.9, 1.0),   # theta=0
        svg3d.view.Viewport(-1.0, -0.5, 0.9, 1.0),   # theta=90
        svg3d.view.Viewport(0.0, -0.5, 0.9, 1.0),    # theta=180
        svg3d.view.Viewport(1.0, -0.5, 0.9, 1.0),    # theta=270
    ]

    render_grid(views, f"{OUTPUT_DIR}/ortho_theta_grid.svg", positions, size=(2048, 512))


def generate_elevation():
    """Generate 4 views showing different elevation angles."""
    scene = create_scene()

    elevations = [0, 30, 60, 90]
    views = []

    for elevation in elevations:
        view = svg3d.View.orthographic(
            scene=scene,
            scene_width=3.0,
            aspect_ratio=1.0,
            theta=45.0,
            elevation=elevation,
        )
        views.append(view)

    # Side by side (1x4)
    positions = [
        svg3d.view.Viewport(-2.0, -0.5, 0.9, 1.0),   # elev=0
        svg3d.view.Viewport(-1.0, -0.5, 0.9, 1.0),   # elev=30
        svg3d.view.Viewport(0.0, -0.5, 0.9, 1.0),    # elev=60
        svg3d.view.Viewport(1.0, -0.5, 0.9, 1.0),    # elev=90
    ]

    render_grid(views, f"{OUTPUT_DIR}/ortho_elevation_grid.svg", positions, size=(2048, 512))


def generate_scene_width():
    """Generate 3 views showing different scene_width values (zoom levels)."""
    scene = create_scene()

    widths = [2.0, 4.0, 8.0]
    views = []

    for width in widths:
        view = svg3d.View.orthographic(
            scene=scene,
            scene_width=width,
            aspect_ratio=1.0,
            theta=45.0,
            elevation=35.264,
        )
        views.append(view)

    # Side by side
    positions = [
        svg3d.view.Viewport(-0.5, -0.25, 0.3, 0.5),
        svg3d.view.Viewport(-0.1, -0.25, 0.3, 0.5),
        svg3d.view.Viewport(0.3, -0.25, 0.3, 0.5),
    ]

    render_grid(views, f"{OUTPUT_DIR}/ortho_scene_width.svg", positions, size=(768, 384))


def generate_axonometric_comparison():
    """Generate side-by-side comparison of isometric, dimetric, trimetric."""
    scene = create_scene()

    iso = svg3d.View.isometric(scene, scene_width=3.0)
    dim = svg3d.View.dimetric(scene, scene_width=3.0)
    tri = svg3d.View.trimetric(scene, scene_width=3.0)

    views = [iso, dim, tri]

    # Side by side
    positions = [
        svg3d.view.Viewport(-0.5, -0.25, 0.3, 0.5),
        svg3d.view.Viewport(-0.1, -0.25, 0.3, 0.5),
        svg3d.view.Viewport(0.3, -0.25, 0.3, 0.5),
    ]

    render_grid(views, f"{OUTPUT_DIR}/ortho_axonometric.svg", positions, size=(768, 384))


def generate_custom_views():
    """Generate comparison of custom elevation views."""
    scene = create_scene()

    # High elevation (emphasize top), low elevation (emphasize front), mid
    view_high = svg3d.View.orthographic(
        scene=scene,
        scene_width=3.0,
        aspect_ratio=1.0,
        theta=45.0,
        elevation=70.0,
    )

    view_low = svg3d.View.orthographic(
        scene=scene,
        scene_width=3.0,
        aspect_ratio=1.0,
        theta=45.0,
        elevation=15.0,
    )

    view_mid = svg3d.View.orthographic(
        scene=scene,
        scene_width=3.0,
        aspect_ratio=1.0,
        theta=45.0,
        elevation=30.0,
    )

    views = [view_high, view_low, view_mid]

    positions = [
        svg3d.view.Viewport(-0.5, -0.25, 0.3, 0.5),
        svg3d.view.Viewport(-0.1, -0.25, 0.3, 0.5),
        svg3d.view.Viewport(0.3, -0.25, 0.3, 0.5),
    ]

    render_grid(views, f"{OUTPUT_DIR}/ortho_custom_views.svg", positions, size=(768, 384))


# =============================================================================
# Main
# =============================================================================

def main():
    print("Generating orthographic view example images...")
    print()

    print("1. Theta rotation grid")
    generate_theta_rotation()

    print("2. Elevation grid")
    generate_elevation()

    print("3. Scene width comparison")
    generate_scene_width()

    print("4. Axonometric comparison")
    generate_axonometric_comparison()

    print("5. Custom views comparison")
    generate_custom_views()

    print()
    print("Done! Images saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
