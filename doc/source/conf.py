# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.append(os.path.join("..", "..", "svg3d"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "svg3d"
copyright = "2024, Jenna Bradley"
author = "Jenna Bradley"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org", None),
    "coxeter": ("https://coxeter.readthedocs.io/en/stable", None),
    "svgwrite": ("https://svgwrite.readthedocs.io/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "svg3d-logo-text.svg",
    "dark_logo": "svg3d-logo-light.svg",
    "top_of_page_buttons": ["view", "edit"],
    "navigation_with_keys": True,
    "dark_css_variables": {
        "color-brand-primary": "#AFA8B9",
        "color-brand-content": "#AFA8B9",
    },
    "light_css_variables": {
        "color-brand-primary": "#4A4453",
        "color-brand-content": "#4A4453",
    },
}
html_favicon = "svg3d-logo.svg"

autodoc_default_options = {
    "members": True,
    "private-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
