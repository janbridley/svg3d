import numpy as np

DEFAULT_LIGHT = np.array([2, 2, 1]) / 2


def hex2rgb(hexc):
    hexc = hexc.lstrip("#")
    return np.array([int(hexc[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0


def rgb2hex(rgb):
    rgb = (rgb * 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*rgb).upper()


def _apply_shading(base_color, shading, factor=0.5):
    # `shading` is a value between -1 and 1
    # factor controls how much lighter/darker we go from the base color
    base_rgb = hex2rgb(base_color)
    shaded_color = base_rgb + factor * shading * (np.ones(3) - base_rgb)

    shaded_color = np.clip(shaded_color, 0, 1)  # Ensure RGB values are within [0, 1]
    return rgb2hex(shaded_color)


def diffuse_lighting(
    face_index, mesh, light_direction=None, base_style=None, base_color="#71618D"
):
    base_style = base_style if base_style is not None else {}
    light_direction = light_direction if light_direction is not None else DEFAULT_LIGHT

    normal = mesh.normals[face_index] / np.linalg.norm(mesh.normals[face_index])
    shading = np.dot(normal, light_direction)

    new_color = _apply_shading(base_color, shading, factor=0.6)

    return base_style | {"fill": new_color}
