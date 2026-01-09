# Changelog

The format is based on `Keep a Changelog <http://keepachangelog.com/en/1.0.0/>` **. This
project adheres to `Semantic Versioning <http://semver.org/spec/v2.0.0.html>`**.

## v0.2.0

### Added

- `UniformShader` object, which applies a flat uniform color to all faces

### Changed

- `Mesh` objects no longer take a style dict, and instead rely on a `Shader` for color

### Removed

- Support for Python 3.10

### Fixed

- Divide-by-zero issue when normalizing mesh normals
- Doc example that used a previous style of flat shading

## v0.1.2

### Fixed

- Image links in README.rst

## v0.1.1

### Added

- Added public ReadtheDocs page.
- Additional examples

## v0.1.0

### Changed:

- Removed `pyrr` dependency by implementing matrix methods
- `Mesh` class is now more verstaile, and supports initialization from `Coxeter` objects

### Added:

- Added `isometric`, `dimetric`, and `trimetric` convenience functions to `View` class.
- Support meshes with nonuniform degree (e.g. mixed tri-quad meshes). Faces are padded
  out to the largest polygon's size in the projection step, and are filtered back down
  before drawing the SVG.
- `get_lookat_matrix` and `get_projection_matrix` methods to replace calls to `pyrr`
  required by the old version
- Additional tests and CI
- New `Shader` submodule, which supports a few basic lighting models and provides an
  interface for future development

### Removed:

- `Camera` class has been replaced with `view.look_at` and `view.projection`
- `Scene` class has been removed in favor of basic Python iterables.
- `extras` from original svg3d
