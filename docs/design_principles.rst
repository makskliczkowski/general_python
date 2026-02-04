Design Principles
=================

**General Python Utilities** is built on a few key architectural principles designed to support high-performance scientific computing and quantum physics simulations.

Backend Agnosticism
-------------------

A primary goal of the library is to write code once and run it on multiple backends. Most core modules, especially `algebra` and `ml`, are designed to be backend-agnostic.

- **NumPy**: Used for standard CPU execution, prototyping, and compatibility with the vast Python scientific ecosystem.
- **JAX**: Used for high-performance execution on GPUs/TPUs, automatic differentiation (AD), and just-in-time (JIT) compilation.

The library achieves this through:
1.  **Unified Interfaces**: Abstract base classes (e.g., `Solver`, `Preconditioner`) that define behavior independent of the underlying array library.
2.  **Backend Dispatch**: Utility functions in `algebra.utils` that return the appropriate array module (`numpy` or `jax.numpy`) based on configuration or input data types.

Modular Architecture
--------------------

The library is organized into distinct top-level packages, each serving a specific scientific domain:

- **`algebra`**: Linear algebra, solvers, and random number generation.
- **`physics`**: Quantum mechanics, statistical physics, and thermodynamics.
- **`lattices`**: Geometry and topology of physical lattices.
- **`ml`**: Machine learning models and optimization (Flax/JAX integration).
- **`maths`**: General mathematical utilities and statistics.
- **`common`**: Shared utilities like plotting, I/O, and logging.

Each module is designed to be as independent as possible, though `physics` often relies on `algebra` and `lattices`.

Lazy Loading
------------

To ensure fast startup times—especially important when running large batches of jobs on a cluster—the library extensively uses **lazy loading**.

- **Heavy Dependencies**: Libraries like JAX, TensorFlow, or large submodules are not imported until they are actually accessed.
- **Implementation**: The `__init__.py` files in top-level packages (e.g., `algebra`, `maths`) use a custom `LazyImporter` class.

**Implication for Users**:
You can import the top-level package without paying the cost of loading the entire ecosystem.

.. code-block:: python

    import general_python.algebra as alg
    # 'solvers' submodule is not loaded yet

    solver = alg.choose_solver(...)
    # Now 'solvers' and necessary backends are loaded

Scientific Rigor
----------------

As a library rooted in quantum physics research:
- **Numerical Stability**: Algorithms are chosen and implemented with stability in mind (e.g., careful handling of log-sum-exp in statistics).
- **Correctness**: Physical units, boundary conditions (PBC/OBC), and symmetries are first-class citizens in modules like `lattices`.
- **Reproducibility**: Random number generation is handled via `algebra.ran_wrapper` to ensure reproducible results across backends.
