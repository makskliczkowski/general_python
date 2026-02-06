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

**Backend RNG and SciPy Helpers**

If you want a backend-specific RNG or SciPy module, request them explicitly.

.. code-block:: python

    from general_python.algebra import utils

    # NumPy backend with RNG + SciPy helpers
    xp, (rng, _), sp    = utils.get_backend("numpy", random=True, scipy=True, seed=123)
    x                   = rng.normal(size=5)
    print(f"Norm via SciPy: {sp.linalg.norm(x):.4f}")

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

**Direct (Backend) Solver**

For small dense systems, the direct backend solve is convenient.

.. code-block:: python

    from general_python.algebra.solvers import choose_solver, SolverType
    import numpy as np

    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 0.0])

    solver_direct   = choose_solver(SolverType.BACKEND, a=A)
    res_direct      = solver_direct.solve_instance(b=b)
    print(f"Direct solution: {res_direct.x}")

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

**Adjacency Matrix and Real-Space Plot**

.. code-block:: python

    A = lat.adjacency_matrix(sparse=True)
    print(f"Adjacency shape: {A.shape}, nnz: {A.nnz}")

    # Quick scatter plot of the real-space sites
    # fig, ax = lat.plot_real_space(show_indices=True, color="tab:blue")

3. Physics & Quantum States
---------------------------

Utilities for quantum mechanics and statistical physics.

.. code-block:: python

    from general_python.physics import density_matrix
    import numpy as np

    # Create a random quantum state vector
    psi     = np.random.rand(4) + 1j * np.random.rand(4)
    psi    /= np.linalg.norm(psi)

    # Compute reduced density matrix for a 2x2 bipartition
    rho     = density_matrix.rho_numpy(psi, dimA=2, dimB=2)
    
    # Von Neumann entropy from eigenvalues of rho
    evals   = density_matrix.rho_spectrum(rho)
    S       = -np.sum(evals * np.log(evals))
    print(f"Entropy: {S:.4f}")

    # Purity: tr(rho^2)
    purity   = np.real(np.trace(rho @ rho))
    print(f"Purity: {purity:.4f}")

**Schmidt Decomposition**

.. code-block:: python

    # Schmidt decomposition (eigenvalue route)
    evals, vecs, _ = density_matrix.schmidt_numpy(psi, dimA=2, dimB=2, eig=True)
    print(f"Top Schmidt weights: {evals[:3]}")

**Two-Site Reduced Density Matrix**

.. code-block:: python

    # Example on a 4-qubit state (size 16)
    ns      = 4
    psi     = np.random.randn(2**ns) + 1j * np.random.randn(2**ns)
    psi    /= np.linalg.norm(psi)

    rho_ij  = density_matrix.rho_two_sites(psi, site_i=0, site_j=2, ns=ns)
    print(f"Two-site rho shape: {rho_ij.shape}")

4. Random Number Generation
---------------------------

Reproducible random numbers using high-quality generators.

.. code-block:: python

    from general_python.maths import random as rng_mod
    import numpy as np

    # Create a seeded NumPy generator
    rng = np.random.default_rng(12345)

    # Draw a Haar-random unitary from the Circular Unitary Ensemble (CUE)
    U   = rng_mod.CUE_QR(4, rng=rng, simple=False)
    print(f"CUE unitary shape: {U.shape}")
    print(f"Unitary check: {np.allclose(U.conj().T @ U, np.eye(4))}")

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

**Using the Callable Apply Function**

.. code-block:: python

    apply_fn, params    = net.get_apply(use_jax=False)
    y2                  = apply_fn(params, x)
    print(f"Output matches: {np.allclose(y, y2)}")

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

**Benchmark Helper**

.. code-block:: python

    from general_python.common.timer import benchmark, timeit
    import numpy as np

    # Context manager timing
    with benchmark("Vectorized op") as stats:
        _ = np.sum(np.arange(1_000_000))
    print(f"Elapsed (s): {stats.elapsed:.6f}")

    # Functional timing
    def work(n):
        return sum(i * i for i in range(n))

    result, dt = timeit(work, 100_000)
    print(f"Timeit elapsed (s): {dt:.6f}")
