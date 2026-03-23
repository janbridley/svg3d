#!/usr/bin/env python3
"""
Generate isometric, dimetric, and trimetric view SVGs for the quickstart example.

This code mirrors the quickstart example in README.rst. When updating the README,
update this file to match.
"""

import os

from coxeter.families import ArchimedeanFamily

import svg3d

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "source", "_static")

# Style from README quickstart example
STYLE = {
    "fill": "#71618D",
    "fill_opacity": "0.85",
    "stroke": "black",
    "stroke_linejoin": "round",
    "stroke_width": "0.005",
}


def main():
    print("Generating iso/dim/tri views from quickstart example...")

    truncated_cube = ArchimedeanFamily.get_shape("Truncated Cube")

    scene = [
        svg3d.Mesh.from_coxeter(
            truncated_cube, shader=svg3d.shaders.DiffuseShader.from_style_dict(STYLE)
        )
    ]

    # Convenience views: isometric, dimetric, and trimetric (all orthographic)
    iso = svg3d.View.isometric(scene, scene_width=2.0)
    dim = svg3d.View.dimetric(scene, scene_width=2.0)
    tri = svg3d.View.trimetric(scene, scene_width=2.0)

    for view, view_type in zip([iso, dim, tri], ["iso", "dim", "tri"], strict=False):
        output_path = os.path.join(OUTPUT_DIR, f"{view_type}.svg")
        svg3d.Engine([view]).render(output_path)
        print(f"  Wrote {output_path}")

    print("Done!")


if __name__ == "__main__":
    main()
