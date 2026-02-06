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
    solver_cg   = choose_solver(SolverType.CG, a=A)
    res_cg      = solver_cg.solve_instance(b=b)
    print(f"CG converged: {res_cg.converged}, Iterations: {res_cg.iterations}")

    # 2. Using MINRES (for symmetric indefinite matrices)
    # You can also pass a matvec function/LinearOperator
    def matvec(v):
        return A @ v

    solver_minres   = choose_solver(SolverType.MINRES, matvec_func=matvec)
    x0              = np.zeros_like(b)
    res_minres      = solver_minres.solve_instance(b=b, x0=x0)
    print(f"MINRES residual norm: {res_minres.residual_norm}")

**Preconditioners**

.. code-block:: python

    from general_python.algebra.preconditioners import choose_precond

    # Create a Jacobi preconditioner (set up with matrix A)
    M = choose_precond("jacobi")
    M.set(A)

    # Use it in a solver
    solver_precond = choose_solver(SolverType.CG, a=A)
    solver_precond.solve_instance(b=b, precond=M)

2. Lattice Geometries
---------------------

Create and navigate lattice structures for physics simulations.

.. code-block:: python

    from general_python.lattices import SquareLattice, HexagonalLattice
    
    # Create a 4x4 square lattice with Periodic Boundary Conditions (PBC)
    lat = SquareLattice(4, 4, bc='pbc')

    print(f"Total sites: {lat.Ns}")
    
    # Get neighbors of site index 0
    neighbors = lat.get_nn(0)
    print(f"Neighbors of site 0: {neighbors}")
    
    # Get coordinates of site (2, 2)
    site    = lat.site_index(2, 2, 0)
    coord   = lat.get_coordinates(site)
    print(f"Coordinates of (2,2): {coord}")

    # Plot the lattice (requires matplotlib)
    # fig, ax = lat.plot_structure(show_indices=True)

3. Physics & Quantum States
---------------------------

Utilities for quantum mechanics and statistical physics.

.. code-block:: python

    from general_python.physics import density_matrix
    import numpy as np

    # Create a random quantum state vector
    psi = np.random.rand(4) + 1j * np.random.rand(4)
    psi /= np.linalg.norm(psi)

    # Compute reduced density matrix for a 2x2 bipartition
    rho     = density_matrix.rho_numpy(psi, dimA=2, dimB=2)
    
    # Von Neumann entropy from eigenvalues of rho
    evals   = density_matrix.rho_spectrum(rho)
    S       = -np.sum(evals * np.log(evals))
    print(f"Entropy: {S:.4f}")

    # Purity: tr(rho^2)
    purity   = np.real(np.trace(rho @ rho))
    print(f"Purity: {purity:.4f}")

4. Random Number Generation
---------------------------

Reproducible random numbers using high-quality generators.

.. code-block:: python

    from general_python.maths import random as rng_mod

    # Create a seeded NumPy generator
    rng = np.random.default_rng(12345)

    # Draw a Haar-random unitary from the Circular Unitary Ensemble (CUE)
    U   = rng_mod.CUE_QR(4, rng=rng)
    print(f"CUE unitary shape: {U.shape}")

5. Machine Learning (Neural Networks)
-------------------------------------

Define and use neural networks with backend flexibility.

.. code-block:: python

    from general_python.ml.networks import choose_network
    import numpy as np

    # Create a simple feed-forward network (NumPy backend)
    net = choose_network(
        "simple",
        input_shape=(16,),
        layers=(8, 4),
        act_fun=("tanh",),
        backend="numpy",
        dtype=np.float32,
    )

    x = np.random.randn(5, 16).astype(np.float32)
    y = net.apply_np(x)
    print(f"Network output shape: {y.shape}")

6. Common Utilities
-------------------

.. code-block:: python

    from general_python.common import Directories, Timer
    
    # Directory management
    dirs = Directories("experiment_data")
    dirs.mkdir()
    
    # Timing code execution
    with Timer("Heavy Computation", verbose=True):
        # simulate work
        _ = [i**2 for i in range(100000)]
    
    # Timer automatically logs the duration
