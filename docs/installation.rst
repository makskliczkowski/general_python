Installation
============

The recommended way to install **GenUtils Python** is via pip or by cloning the repository directly from GitHub.

Prerequisites
-------------
- Python 3.10+
- pip
- (Optional) JAX/Flax for machine learning and accelerated linear algebra
- (Optional) TensorFlow/Keras for certain ML integrations

Install from Source (GitHub)
----------------------------

Since this package is hosted on GitHub, you can install it directly or clone and install.

**1. Standard Development Installation (Recommended)**

If you plan to modify the code or contribute:

.. code-block:: bash

    git clone https://github.com/makskliczkowski/QuantumEigenSolver.git
    cd QuantumEigenSolver/pyqusolver/Python
    pip install -e ".[standard,dev,docs,ml]"

This installs the package in editable mode along with the recommended JAX/Flax stack, development tools, documentation tools, and ML utilities.

**2. Minimal Installation**

To simply use the library:

.. code-block:: bash

    pip install .

**3. Standard Runtime Installation**

.. code-block:: bash

    pip install ".[standard]"

The ``.[jax]`` extra remains available as a compatible alias for the same JAX/Flax-enabled stack.

Dependencies
------------

Core dependencies (automatically installed):
- ``numpy``
- ``scipy``
- ``matplotlib``

Optional dependencies:
- ``standard`` / ``jax``: ``jax``, ``jaxlib``, ``flax``, ``optax`` for accelerated linear algebra and neural networks.
- ``pandas``, ``scikit-learn``: For machine learning utilities (via ``[ml]`` extra).
- ``sphinx``, ``sphinx_rtd_theme``: For building documentation (via ``[docs]`` extra).
- ``pytest``, ``black``, ``flake8``: For testing and development (via ``[dev]`` extra).

Troubleshooting
---------------
If you encounter issues with JAX installation (especially on GPUs), please refer to the official `JAX installation guide <https://github.com/google/jax#installation>`_.
