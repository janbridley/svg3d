.. _development:

=================
Development Guide
=================


All contributions to **svg3d** are welcome!
Developers are invited to contribute to the framework by pull request to the package repository on `GitHub`_, and all users are welcome to provide contributions in the form of **user feedback** and **bug reports**.
We recommend discussing new features in form of a proposal on the issue tracker for the appropriate project prior to development.

.. _github: https://github.com/janbridley/svg3d

General Guidelines
==================

All code contributed to **svg3d** must adhere to the following guidelines:

  * Use a two branch model of development:

    - Most new features and bug fixes should be developed in branches based on ``main``.
    - API incompatible changes and those that significantly change existing functionality should be based on ``breaking``
  * Hard dependencies (those that end users must install to use **svg3d**) are discouraged, and should be avoided where possible.
  * All code should adhere to the source code conventions and satisfy the documentation and testing requirements discussed below.


Style Guidelines
----------------

The **svg3d** package adheres to a reasonably strict set of style guidelines.
All code in **svg3d** should be formatted using `ruff`_ via pre-commit. This provides an easy workflow to enforce a number of style and syntax rules that have been configured for the project.

.. tip::

    `pre-commit`_ has been configured to run a number of linting and formatting tools. It is recommended to set up pre-commit to run automatically:

    .. code-block:: bash

        python -m pip install pre-commit
        pre-commit install # Set up git hook scripts

    Alternatively, the tools can be run manually with the following command:

    .. code-block:: bash

        git add .; pre-commit run

.. _ruff: https://docs.astral.sh/ruff/
.. _pre-commit: https://pre-commit.com/


Documentation
-------------

API documentation should be written as part of the docstrings of the package in the `Google style <https://google.github.io/styleguide/pyguide.html#383-functions-and-methods>`__.

Docstrings are automatically validated using `pydocstyle <http://www.pydocstyle.org/>`_ whenever the ruff pre-commit hooks are run.
The `official documentation <https://svg3d.readthedocs.io/>`_ is generated from the docstrings using `Sphinx <http://www.sphinx-doc.org/en/stable/index.html>`_.

In addition to API documentation, inline comments are strongly encouraged.
Code should be written as transparently as possible, so the primary goal of documentation should be explaining the algorithms or mathematical concepts underlying the code.

Building Documentation
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd doc
  make clean
  make html
  open build/html/index.html


Unit Tests
----------

All code should include a set of tests which test for correct behavior.
All tests should be placed in the ``tests`` folder at the root of the project.
In general, most parts of svg3d primarily require `unit tests <https://en.wikipedia.org/wiki/Unit_testing>`_, but where appropriate `integration tests <https://en.wikipedia.org/wiki/Integration_testing>`_ are also welcome. Core functions should be tested against the sample CIF files included in ``tests/sample_data``.
Tests in **svg3d** use the `pytest <https://docs.pytest.org/>`__ testing framework.
To run the tests, simply execute ``pytest`` at the root of the repository.
