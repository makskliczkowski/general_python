Usage
=====

This section provides **working, end-to-end examples** for the core components of the
*General Python Utilities* library. Each snippet is runnable and uses real APIs from
the package.

1. Linear Algebra & Solvers
---------------------------

The algebra stack exposes a unified NumPy/JAX backend, deterministic RNG helpers,
and a flexible solver interface (direct, iterative, matrix-free, and preconditioned).

**Backend Selection, RNG, and SciPy Helpers**

.. code-block:: python

    from general_python.algebra import utils

    # Global backend info
    print(f"Active backend: {utils.ACTIVE_BACKEND_NAME}")

    # Explicit backend selection
    xp  = utils.get_backend("numpy")
    x   = xp.linspace(0.0, 1.0, 5)

    # Reproducible RNG + SciPy helpers (NumPy backend)
    xp, (rng, _), sp = utils.get_backend("numpy", random=True, scipy=True, seed=123)
    A = rng.normal(size=(4, 4))
    A = 0.5 * (A + A.T)  # symmetrize
    evals = sp.linalg.eigvalsh(A)
    print("eigvals:", evals)

    # Optional JAX backend (if installed)
    if utils.JAX_AVAILABLE:
        jxp, (jrn, key), jsp = utils.get_backend("jax", random=True, scipy=True, seed=0)
        key, sub = jrn.split(key)
        v = jrn.normal(sub, shape=(3,))
        print("jax vector:", v)

**Preconditioned Conjugate Gradient (SPD systems)**

.. code-block:: python

    import numpy as np
    from general_python.algebra.solvers import choose_solver, SolverType
    from general_python.algebra.preconditioners import choose_precond

    n = 200
    diag = np.linspace(1.0, 10.0, n)
    A = np.diag(diag)
    b = np.ones(n)

    # Jacobi preconditioner (diagonal inverse)
    M = choose_precond("jacobi")
    M.set(A)

    solver = choose_solver(SolverType.CG, a=A)
    result = solver.solve_instance(b=b, precond=M, tol=1e-10, maxiter=200)
    print(f"CG converged: {result.converged}, iters: {result.iterations}")

**Matrix-Free MINRES (symmetric indefinite systems)**

.. code-block:: python

    def matvec(v):
        return A @ v

    solver = choose_solver(SolverType.MINRES, matvec_func=matvec)
    x0 = np.zeros_like(b)
    result = solver.solve_instance(b=b, x0=x0)
    print(f"MINRES residual norm: {result.residual_norm}")

**Direct Backend Solver (dense systems)**

.. code-block:: python

    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 0.0])

    solver = choose_solver(SolverType.BACKEND, a=A, sigma=1e-12)
    result = solver.solve_instance(b=b)
    print("Direct solution:", result.x)

2. Lattice Geometries
---------------------

The lattice module provides geometry-aware models with real/reciprocal vectors,
neighbor maps, and plotting helpers. Use the factory to stay backend-agnostic.

**Factory + Metadata**

.. code-block:: python

    from general_python.lattices import choose_lattice, available_lattices

    print("Available lattices:", available_lattices())

    lat = choose_lattice("square", lx=4, ly=4, bc="pbc")
    print(lat.summary_string())

    # Coordinates and nearest neighbors
    site = lat.site_index(2, 2, 0)
    coord = lat.get_coordinates(site)
    nn = lat.get_nn(site)
    print(f"Site {site} coord: {coord}, nn: {nn}")

**Adjacency + Real-Space Overview**

.. code-block:: python

    A = lat.adjacency_matrix(sparse=True, include_nnn=True)
    print(f"Adjacency shape: {A.shape}, nnz: {A.nnz}")

    # Compact report with vectors and Brillouin zone
    print(lat.describe(max_rows=6))

**Plotting (matplotlib)**

.. code-block:: python

    # fig, ax = lat.plot.structure(show_indices=True)
    # fig, ax = lat.plot.real_space(show_indices=True, color="tab:blue")

3. Physics & Quantum States
---------------------------

Density-matrix utilities cover reduced states, spectra, and entanglement
metrics with NumPy implementations.

**Reduced Density Matrix + Entropy**

.. code-block:: python

    import numpy as np
    from general_python.physics import density_matrix, entropy

    # Random 2-qubit pure state
    psi = np.random.randn(4) + 1j * np.random.randn(4)
    psi /= np.linalg.norm(psi)

    rho = density_matrix.rho_numpy(psi, dimA=2, dimB=2)
    evals = density_matrix.rho_spectrum(rho)
    S = -np.sum(evals * np.log(evals))
    print(f"Von Neumann entropy: {S:.6f}")

    # Compare to Page value (finite-size average)
    S_page = entropy.EntropyPredictions.Mean.page(da=2, db=2)
    print(f"Page value (dA=2, dB=2): {S_page:.6f}")

**Schmidt Decomposition + Two-Site RDM**

.. code-block:: python

    # Schmidt decomposition via eigen-decomposition
    vals, vecs, _ = density_matrix.schmidt_numpy(psi, dimA=2, dimB=2, eig=True)
    print("Top Schmidt weights:", vals[:3])

    # Two-site reduced density matrix for a 4-qubit state
    ns = 4
    psi = np.random.randn(2**ns) + 1j * np.random.randn(2**ns)
    psi /= np.linalg.norm(psi)

    rho_ij = density_matrix.rho_two_sites(psi, site_i=0, site_j=2, ns=ns)
    print("Two-site rho shape:", rho_ij.shape)

4. Random Number Generation
---------------------------

High-quality random matrix samplers for quantum information tasks.

**Haar-Random Unitary (CUE)**

.. code-block:: python

    import numpy as np
    from general_python.maths import random as rng_mod

    rng = np.random.default_rng(12345)
    U = rng_mod.CUE_QR(4, rng=rng, simple=False)

    # Unitarity check
    I = np.eye(4)
    print("Unitary check:", np.allclose(U.conj().T @ U, I))

    # Eigenvalue phases on the unit circle
    phases = np.angle(np.linalg.eigvals(U))
    print("Eigenphases:", phases)

5. Machine Learning (Neural Networks)
-------------------------------------

Networks are created through a single factory that supports NumPy and JAX
backends. The examples below use the lightweight `simple` network.

**NumPy Backend (quick experiments)**

.. code-block:: python

    import numpy as np
    from general_python.ml.networks import choose_network

    net = choose_network(
        "simple",
        input_shape=(16,),
        layers=(32, 16),
        act_fun=("tanh", "tanh"),
        backend="numpy",
        dtype=np.float32,
    )

    x = np.random.randn(4, 16).astype(np.float32)
    y = net.apply_np(x)
    print("Output shape:", y.shape)
    print("Total params:", sum(net.nparams))

    apply_fn, params = net.get_apply(use_jax=False)
    y2 = apply_fn(params, x)
    print("Output matches:", np.allclose(y, y2))

**Optional JAX Backend (accelerated)**

.. code-block:: python

    from general_python.algebra import utils

    if utils.JAX_AVAILABLE:
        net_jax = choose_network(
            "simple",
            input_shape=(16,),
            layers=(32, 16),
            act_fun=("tanh", "tanh"),
            backend="jax",
            dtype="float32",
        )
        jxp = utils.get_backend("jax")
        xj = jxp.ones((2, 16))
        yj = net_jax.apply_jax(xj)
        print("JAX output shape:", yj.shape)

6. Common Utilities
-------------------

Convenience utilities for directories, timing, and lightweight benchmarking.

**Directories**

.. code-block:: python

    from general_python.common import Directories

    base = Directories("experiment_data").mkdir()
    run_dir = base.join("run_001", create=True)

    # List files (empty at first)
    print(run_dir.list_files())

**Timing & Benchmarking**

.. code-block:: python

    from general_python.common import Timer
    from general_python.common.timer import benchmark, timeit
    import numpy as np

    # Context manager timing
    with Timer("Heavy Computation", verbose=True):
        _ = [i**2 for i in range(100000)]

    # Benchmark helper
    with benchmark("Vectorized op") as stats:
        _ = np.sum(np.arange(1_000_000))
    print(f"Elapsed (s): {stats.elapsed:.6f}")

    # Functional timing
    def work(n):
        return sum(i * i for i in range(n))

    result, dt = timeit(work, 100_000)
    print(f"Timeit elapsed (s): {dt:.6f}")
