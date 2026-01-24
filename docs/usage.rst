Usage
=====

This section illustrates usage examples for the key components of the *General Python Utilities* library.

1. Linear Algebra & Solvers
---------------------------

The library provides a unified interface for linear algebra that works with both NumPy and JAX.

**Backend Management**

You can switch backends or let the library detect the best available one.

.. code-block:: python

    import numpy as np
    from general_python.algebra import utils

    # Check active backend (defaults to 'numpy' or 'jax' if available)
    print(f"Active backend: {utils.ACTIVE_BACKEND_NAME}")

    # Explicitly get backend operations
    xp = utils.get_backend("numpy")
    arr = xp.array([1, 2, 3])

**Using Iterative Solvers**

Solve :math:`Ax = b` using Krylov subspace methods like CG or MINRES.

.. code-block:: python

    from general_python.algebra.solvers import choose_solver, SolverType
    import numpy as np

    # Define a symmetric positive-definite matrix A and vector b
    N = 100
    A = np.diag(np.arange(1, N + 1))
    b = np.ones(N)

    # 1. Using Conjugate Gradient (CG)
    # You can pass the matrix directly
    solver_cg = choose_solver(SolverType.CG, A=A)
    res_cg = solver_cg.solve(b)
    print(f"CG converged: {res_cg.success}, Iterations: {res_cg.info['iter']}")

    # 2. Using MINRES (for symmetric indefinite matrices)
    # You can also pass a matvec function/LinearOperator
    def matvec(v):
        return A @ v

    solver_minres = choose_solver(SolverType.MINRES, matvec=matvec, shape=(N, N))
    res_minres = solver_minres.solve(b)
    print(f"MINRES solution norm: {np.linalg.norm(res_minres.x)}")

**Preconditioners**

.. code-block:: python

    from general_python.algebra.preconditioners import choose_precond

    # Create an ILU preconditioner
    M = choose_precond("ilu", A=A)

    # Use it in a solver
    solver_precond = choose_solver(SolverType.CG, A=A, M=M)
    solver_precond.solve(b)

2. Lattice Geometries
---------------------

Create and navigate lattice structures for physics simulations.

.. code-block:: python

    from general_python.lattices import SquareLattice, HexagonalLattice
    
    # Create a 4x4 square lattice with Periodic Boundary Conditions (PBC)
    lat = SquareLattice(4, 4, bc='pbc')

    print(f"Total sites: {lat.Ns}")
    
    # Get neighbors of site index 0
    neighbors = lat.get_neighbors(0)
    print(f"Neighbors of site 0: {neighbors}")
    
    # Get coordinates of site (2, 2)
    coord = lat.get_coord((2, 2))
    print(f"Coordinates of (2,2): {coord}")

    # Plot the lattice (requires matplotlib)
    # lat.plot_lattice(show=True)

3. Physics & Quantum States
---------------------------

Utilities for quantum mechanics and statistical physics.

.. code-block:: python

    from general_python.physics import density_matrix, entropy
    import numpy as np

    # Create a random quantum state vector
    psi = np.random.rand(4) + 1j * np.random.rand(4)
    psi /= np.linalg.norm(psi)

    # Compute density matrix rho = |psi><psi|
    rho = density_matrix.create_density_matrix(psi)
    
    # Calculate Von Neumann Entropy
    # S = -tr(rho ln rho)
    S = entropy.von_neumann_entropy(rho)
    print(f"Entropy: {S:.4f}")

    # Calculate Purity
    # P = tr(rho^2)
    purity = entropy.purity(rho)
    print(f"Purity: {purity:.4f}")

4. Random Number Generation
---------------------------

Reproducible random numbers using high-quality generators.

.. code-block:: python

    from general_python.maths import random as rng_mod

    # Create a generator with a specific seed
    rng = rng_mod.Xoshiro256(seed=12345)

    # Generate random numbers
    r = rng.random()
    print(f"Random float: {r}")
    
    # Generate random integers
    ints = rng.randint(0, 10, size=5)
    print(f"Random integers: {ints}")

5. Machine Learning (Neural Networks)
-------------------------------------

Define and use neural networks with backend flexibility.

.. code-block:: python

    from general_python.ml.networks import DenseSymm
    import jax.numpy as jnp
    import jax

    # Define a dense symmetric network (RBM-like)
    # Note: Requires JAX backend for this specific network
    
    net = DenseSymm(input_size=16, hidden_size=4, output_size=1)
    
    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = net.init(key, jnp.ones((1, 16)))
    
    # Forward pass
    output = net.apply(params, jnp.ones((5, 16)))
    print(f"Network output shape: {output.shape}")

6. Common Utilities
-------------------

.. code-block:: python

    from general_python.common import Directories, Timer
    
    # Directory management
    dirs = Directories("experiment_data")
    dirs.create_if_not_exists()
    
    # Timing code execution
    with Timer("Heavy Computation"):
        # simulate work
        _ = [i**2 for i in range(100000)]
    
    # Timer automatically logs the duration
