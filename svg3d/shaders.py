from abc import ABC, abstractmethod

import numpy as np

DEFAULT_LIGHT = np.array([1, 1, 0.5], dtype=float)


def hex2rgb(hexc):
    """
    Convert a hexadecimal color string to an RGB array normalized to [0, 1].

    Parameters
    ----------
    hexc : str
        A hexadecimal color string, with or without a leading `#`.

    Returns
    -------
     :math:`(3,)` :class:`numpy.ndarray`:
        A NumPy array containing RGB values normalized to the range [0, 1].

    Examples
    --------
    >>> hex2rgb("#FFFFFF")
    array([1., 1., 1.])
    >>> hex2rgb("000000")
    array([0., 0., 0.])
    """
    hexc = hexc.lstrip("#")
    return np.array([int(hexc[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0


def rgb2hex(rgb):
    """
    Convert an RGB color array to a hexadecimal color string.

    Parameters
    ----------
    rgb : :math:`(3,)` :class:`numpy.ndarray`: The RGB values to convert.

    Returns
    -------
    str : A hexadecimal color string in uppercase format, prefixed with `#`.

    Examples
    --------
    >>> rgb2hex(np.array([1.0, 1.0, 1.0]))
    '#FFFFFF'
    >>> rgb2hex(np.array([0.0, 0.0, 0.0]))
    '#000000'
    """
    rgb = (rgb * 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*rgb).upper()


def _apply_shading(base_color, shading, absorbance=0.5):
    # `shading` is a value between -1 and 1
    # factor controls how much lighter/darker we go from the base color
    base_rgb = hex2rgb(base_color)
    shaded_color = base_rgb + absorbance * shading * (np.ones(3) - base_rgb)

    shaded_color = np.clip(shaded_color, 0, 1)  # Ensure RGB values are within [0, 1]
    return rgb2hex(shaded_color)


def diffuse_lighting(
    face_index, mesh, light_direction=None, base_style=None, base_color="#71618D"
):
    """Apply Lambertian (dot product diffuse) shading to a face in an \
    :obj:`~.Mesh`.

    This is a convenience function for backwards compatibility. The full-featured
    :obj:`~.Shader` class should be used in most instances.
    """

    base_style = base_style if base_style is not None else {}
    light_direction = light_direction if light_direction is not None else DEFAULT_LIGHT

    normal = mesh.normals[face_index] / np.linalg.norm(mesh.normals[face_index])
    shading = np.dot(normal, light_direction)

    new_color = _apply_shading(base_color, shading, absorbance=0.6)

    return base_style | {"fill": new_color}


class Shader(ABC):
    """
    Abstract base class for shaders.
    """

    def __init__(self, base_color="#71618D", base_style=None):
        """Initialize the shader.

        Parameters
        ----------
        base_color : str, optional
            A hexadecimal-formatted color string for the mesh. Default is "#71618D".
        base_style: dict | None, optional
            The style attribute dict for the :obj:`~.Shader`.
        """

        self._base_color = base_color
        self._base_style = base_style

    @abstractmethod
    def __call__(self, face_index, mesh, absorbance=0.6):
        """Compute the shaded style for a face in a mesh.

        Abstract method to be implemented in subclasses.
        """
        return {}

    @property
    def base_style(self):
        """dict: Get or set the style attribute dict for the object."""
        return self._base_style

    @base_style.setter
    def base_style(self, base_style: dict):
        self._base_style = base_style


class DiffuseShader(Shader):
    """
    Shade Mesh objects with per-face, Lambertian (dot product diffuse) lighting.
    """

    def __init__(
        self, base_color="#71618D", light_direction=DEFAULT_LIGHT, base_style=None
    ):
        """Initialize the diffuse shader.

        Parameters
        ----------
        base_color : str, optional
            A hexadecimal-formatted color string for the mesh. Default is "#71618D".
        light_direction : iterable of float, optional
            A 3-element array specifying the direction of the light source.
            Default is (1.0, 1.0, 0.5).
        base_style : dict | None, optional
            The style dict for the :obj:`~.Shader`.
        """
        super().__init__(base_color=base_color, base_style=base_style)
        self._diffuse_light_direction = np.asarray(light_direction)

    @classmethod
    def from_style_dict(cls, style: dict, light_direction=DEFAULT_LIGHT):
        """Create a :obj:`~.Shader` instance with a style dictionary.

        Parameters
        ----------
        style : dict
            The style dict for the :obj:`~.Shader`
        light_direction : array or list of float, optional.
            A 3-element iterable specifying the diffuse light direction. Default \
            value: (1.0, 1.0, 0.5)
        """
        new = cls(base_color=style["fill"], light_direction=light_direction)
        new.base_style = style
        return new

    @classmethod
    def from_color(cls, base_color):
        """Create a :obj:`~.Shader` instance with a specified base color.

        Parameters
        ----------
        base_color : str
            The base color as a hexadecimal string (e.g., `#FFFFFF`).
        """
        return cls(base_color=base_color)

    @classmethod
    def from_color_and_direction(cls, base_color, light_direction):
        """Create a :obj:`~.Shader` instance with a specified base color \
        and light direction.

        Parameters
        ----------
        base_color : str
            The base color as a hexadecimal string (e.g., `#FFFFFF`).
        light_direction : array or list of float
            A 3-element iterable specifying the diffuse light direction.
        """
        return cls(base_color=base_color, light_direction=light_direction)

    def __call__(self, face_index, mesh, absorbance=0.6):
        """Compute the shaded style for a face in a mesh.

        Parameters
        ----------
        face_index : int
            Index of the face in the mesh.
        mesh : Mesh
            An svg3d mesh object.
        absorbance : float, optional
            The "absorbance" of the mesh surface. Should fall in the range [0.0, 1.0), \
            with larger values equating to darker shading. Default is 0.6.

        Returns
        -------
        dict
            A dictionary containing the SVG style attributes for the shaded face.
        """
        base_style = self.base_style if self.base_style is not None else {}

        normal = mesh.normals[face_index] / np.linalg.norm(mesh.normals[face_index])
        shading = np.dot(normal, self.diffuse_light_direction)

        new_color = self._apply_shading(self.base_color, shading, absorbance=absorbance)

        return {**base_style, "fill": new_color}

    def _apply_shading(self, base_color, shading, absorbance=0.5):
        """Apply shading model to an input color."""
        base_rgb = hex2rgb(base_color)
        shaded_color = base_rgb + absorbance * shading * (np.ones(3) - base_rgb)
        shaded_color = np.clip(shaded_color, 0, 1)
        return rgb2hex(shaded_color)

    @property
    def diffuse_light_direction(self):
        """
        np.ndarray: A 3-element array representing the direction of the light source.
        """
        return self._diffuse_light_direction

    @diffuse_light_direction.setter
    def diffuse_light_direction(self, light_direction):
        """
        Set the direction of the diffuse light source.

        Parameters
        ----------
        light_direction : array or list of float
            A 3-element iterable specifying the diffuse light direction.

        Raises
        ------
        AssertionError
            If light_direction is not an iterable of length three.
        """
        msg = "Light direction should be an iterable with length three."
        assert hasattr(light_direction, "__len__") and len(light_direction) == 3, msg
        self._diffuse_light_direction = np.asarray(light_direction)

    @property
    def base_color(self):
        """dict: Get or set the base color for the mesh from a hexadecimal string."""
        return self._base_color

    @base_color.setter
    def base_color(self, base_color):
        self._base_color = base_color
