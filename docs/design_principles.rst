Design Principles
=================

This project is designed for scientific code that must remain readable,
backend-aware, and numerically explicit.

1. Explicit scientific contracts
--------------------------------

Public modules should document:

- expected input and output shapes
- dtype assumptions and promotion behavior
- numerical stability caveats
- determinism and reproducibility considerations

The goal is to reduce ambiguity in downstream simulation code.

2. Backend-aware by default
---------------------------

- NumPy is the baseline backend used for broad compatibility.
- JAX support is optional and used where acceleration or autodiff is relevant.
- Backend selection should be explicit in experiments and benchmarks to avoid accidental drift.

3. Lazy-loading for import performance
--------------------------------------

Top-level packages expose key symbols while deferring heavy imports until first use.
This keeps startup cost low in scripts and batch jobs.

4. Numerical robustness over convenience
----------------------------------------

When algorithms have known conditioning issues, APIs should prefer:

- stable formulations where practical
- tolerances and regularization controls
- clear documentation of failure modes

5. Minimal, testable interfaces
-------------------------------

- Keep package boundaries clear (`algebra`, `lattices`, `maths`, `ml`, `physics`, `common`).
- Prefer small public entry points plus internal helpers.
- Validate behavior with automated tests and documentation builds.

Practical workflow
------------------

Typical local developer loop:

.. code-block:: bash

    pip install -e ".[dev,docs]"
    pytest
    cd docs && make html

For ML/JAX workflows:

.. code-block:: bash

    pip install -e ".[ml,jax]"
