Getting Started
===============

This page provides the shortest path to install, run, and validate the project.

Installation
------------

The project is configured through ``pyproject.toml`` with optional extras.

Core install
^^^^^^^^^^^^

.. code-block:: bash

    pip install -e .

Development + tests
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install -e ".[dev]"

Documentation build dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install -e ".[docs]"

Machine-learning dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install -e ".[ml]"

Optional JAX-only stack
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install -e ".[jax]"

Backend expectations
--------------------

- **NumPy** is the default and baseline backend across the library.
- **JAX** is optional and used in selected modules for accelerator-enabled workflows.
- Some APIs dispatch by backend availability; for reproducible studies, keep backend choice fixed per run.

Quick verification
------------------

Run core tests:

.. code-block:: bash

    pytest

Run the documentation import check:

.. code-block:: bash

    python test_documentation.py

Build Sphinx docs locally:

.. code-block:: bash

    cd docs
    make html

Output is generated at ``docs/_build/html/index.html``.

Notes on reproducibility
------------------------

- Set explicit RNG seeds in workflows that rely on random initialization.
- Expect minor floating-point differences between NumPy and JAX backends due to execution and reduction ordering.
