Getting Started
===============

This guide will help you install **General Python Utilities** and get started with basic usage and testing.

Installation
------------

The library is designed to be installed as a standard Python package.

**Prerequisites**

- Python 3.8 or higher
- pip

**Standard Installation**

To install the package with core dependencies (NumPy, SciPy, Matplotlib):

.. code-block:: bash

    pip install .

**Development Installation**

If you plan to contribute or run the test suite, install with development dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

**Machine Learning Support**

To enable machine learning features (JAX, Flax, TensorFlow, etc.):

.. code-block:: bash

    pip install ".[ml]"

**Building Documentation**

To build this documentation locally:

.. code-block:: bash

    pip install ".[docs]"
    cd docs
    make html

Backend Configuration
---------------------

One of the core features of `general_python` is its ability to switch between **NumPy** (CPU) and **JAX** (CPU/GPU/TPU) backends seamlessly for linear algebra and array operations.

**Automatic Detection**

The library automatically detects if JAX is installed and available. If JAX is found, it may default to it for certain operations, or you can explicitly control this behavior.

**Explicit Selection**

You can check and select the backend using the `algebra.utils` module.
The `backend_mgr` object allows you to switch the active backend globally for the session.

.. code-block:: python

    from general_python.algebra import utils

    # Check active backend
    print(f"Active backend: {utils.ACTIVE_BACKEND_NAME}")

    # Switch globally to JAX (if available)
    if utils.JAX_AVAILABLE:
        utils.backend_mgr.set_active_backend("jax")

    # Get the backend module explicitly (independent of global setting)
    xp = utils.get_backend('jax')  # Returns jax.numpy

Running Tests
-------------

The project uses `pytest` for testing.

**Run All Tests**

To run the full test suite:

.. code-block:: bash

    pytest

**Run Documentation Tests**

To verify that all modules are importable and documentation is consistent:

.. code-block:: bash

    python3 test_documentation.py

**Troubleshooting**

If you encounter `ImportError` regarding `general_python`, ensure you have installed the package in editable mode (`pip install -e .`) or that your `PYTHONPATH` includes the repository root.
