Usage
=====

This section illustrates basic usage examples for the key components of the *General Python Utilities* library. After installing the package, you can import its modules and use their functions and classes in your Python code.

**1. Linear Algebra Example**

For example, to compute eigenvalues and eigenvectors of a matrix using the linear algebra utilities:

.. code-block:: python

    import numpy as np
    from general_python import linear_algebra as la

    # Define a 2x2 matrix (e.g., Pauli-Z matrix in physics)
    H = np.array([[1, 0],
                  [0, -1]])
    eigenvalues, eigenvectors = la.eig(H)
    print("Eigenvalues:", eigenvalues)

In this example, `la.eig` is a function provided by the library (as a wrapper around NumPy) to compute eigenvalues and eigenvectors. The library's linear algebra module may also provide other operations like solving linear systems or performing matrix decompositions.

**2. Random Number Generation Example**

You can use the library's random number generator to produce reproducible sequences of random numbers. For instance:

.. code-block:: python

    from general_python.random import Xoshiro256

    rng = Xoshiro256(seed=42)
    values = [rng.random() for _ in range(5)]
    print("Random values:", values)

Here, `Xoshiro256` is an example of a pseudorandom number generator class implemented in the library. After seeding the generator, calling `rng.random()` returns random floating-point numbers. This is useful in simulations where you need a reliable and fast RNG beyond Python's built-in `random` module.

**3. Lattice Utility Example**

The library can assist in creating and handling lattice structures. For example, to create a 2D square lattice and query neighboring sites:

.. code-block:: python

    from general_python import lattice

    # Create a 4x4 square lattice and get neighbors of a specific site
    lat = lattice.SquareLattice(4, 4)
    neighbors = lat.get_neighbors((2, 2))
    print("Neighbors of site (2, 2):", neighbors)

In this example, `SquareLattice` is a class that generates a square lattice of the given dimensions, and `get_neighbors` returns adjacent sites for a given lattice coordinate.

These examples demonstrate typical usage patterns. For more detailed information on all available functions and classes, see the API Reference section.
