Installation
============

The recommended way to install **GenUtils Python** is via pip or by cloning the repository directly from GitHub.

Prerequisites
-------------
- Python 3.8+
- pip
- (Optional) JAX/Flax for machine learning and accelerated linear algebra
- (Optional) TensorFlow/Keras for certain ML integrations

Install from Source (GitHub)
----------------------------

Since this package is hosted on GitHub, you can install it directly or clone and install.

**1. Development Installation (Recommended)**

If you plan to modify the code or contribute:

.. code-block:: bash

    git clone https://github.com/makskliczkowski/general_python.git
    cd general_python
    pip install -e ".[dev,docs,ml]"

This installs the package in editable mode along with development, documentation, and machine learning dependencies.

**2. Standard Installation**

To simply use the library:

.. code-block:: bash

    pip install .

Or with specific extras:

.. code-block:: bash

    pip install ".[ml]"

Dependencies
------------

Core dependencies (automatically installed):
- ``numpy``
- ``scipy``
- ``matplotlib``

Optional dependencies:
- ``jax``, ``jaxlib``, ``flax``: For accelerated linear algebra and neural networks.
- ``tensorflow``, ``pandas``, ``scikit-learn``: For machine learning utilities (via ``[ml]`` extra).
- ``sphinx``, ``sphinx_rtd_theme``: For building documentation (via ``[docs]`` extra).
- ``pytest``, ``black``, ``flake8``: For testing and development (via ``[dev]`` extra).

Troubleshooting
---------------
If you encounter issues with JAX installation (especially on GPUs), please refer to the official `JAX installation guide <https://github.com/google/jax#installation>`_.
