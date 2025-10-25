Usage
=====

This section illustrates usage examples for the key components of the *General Python Utilities* library. After installing the package, you can import its modules and use their functions and classes in your Python code.

**1. Backend Management and Linear Algebra**

The library automatically detects and manages NumPy/JAX backends for optimal performance:

.. code-block:: python

    import numpy as np
    from general_python.algebra import utils

    # Get the global backend (automatically detects JAX if available)
    backend = utils.get_global_backend()
    print(f"Active backend: {utils.backend_mgr.name}")

    # Create a matrix and compute eigenvalues
    H = backend.array([[1, 0], [0, -1]])  # Pauli-Z matrix
    eigenvals, eigenvecs = backend.linalg.eigh(H)
    print("Eigenvalues:", eigenvals)

    # Use advanced solvers
    from general_python.algebra.solvers import choose_solver, SolverType
    
    # Set up a linear system Ax = b
    A = backend.array([[4, 1], [1, 3]], dtype=backend.float64)
    b = backend.array([1, 2], dtype=backend.float64)
    
    # Choose and use a solver
    solver = choose_solver(SolverType.CG, A=A)
    result = solver.solve(b)
    print("Solution:", result.x)

**2. High-Quality Random Number Generation**

Use robust random number generators for reproducible scientific computing:

.. code-block:: python

    from general_python.maths import __random__ as RandomMod
    from general_python.algebra.utils import get_global_backend

    # Use high-quality Xoshiro256 generator
    rng = RandomMod.Xoshiro256(seed=42)
    values = [rng.random() for _ in range(5)]
    print("Random values:", values)

    # Or use backend-specific random generation
    backend, random_state = get_global_backend(random=True, seed=123)
    if hasattr(random_state, 'uniform'):  # JAX
        random_vals = random_state.uniform(random_state[1], shape=(5,))
    else:  # NumPy
        random_vals = random_state[0].uniform(size=5)
    print("Backend random values:", random_vals)

**3. Lattice Structures and Visualization**

Create and manipulate lattice geometries for condensed matter physics:

.. code-block:: python

    from general_python.lattices import SquareLattice, HexagonalLattice
    
    # Create a 4x4 square lattice with periodic boundaries
    square_lat = SquareLattice(4, 4, boundary_conditions="periodic")
    neighbors = square_lat.get_neighbors((2, 2))
    print("Neighbors of site (2, 2):", neighbors)
    
    # Create a hexagonal lattice
    hex_lat = HexagonalLattice(3, 3)
    print(f"Total sites: {hex_lat.get_total_sites()}")
    
    # Visualize the lattice (if plotting is available)
    try:
        square_lat.plot_lattice()
    except ImportError:
        print("Plotting not available")

**4. Machine Learning with Flexible Backends**

Use neural networks with automatic JAX/NumPy backend selection:

.. code-block:: python

    from general_python.ml import networks
    from general_python.ml import __general__ as ml_general
    
    # Set up ML parameters
    params = ml_general.MLParams(
        epo=100,         # epochs
        batch=32,        # batch size
        lr=0.001,        # learning rate
        reg={},          # regularization
        loss='mse',      # loss function
        fNum=10,         # feature number
        shape=(10,),     # input shape
        optimizer='adam' # optimizer
    )
    
    # Create a simple neural network
    # (Implementation depends on the specific network class)
    print(f"ML Parameters configured for {params.epo} epochs")

**5. Quantum Physics Utilities**

Perform quantum state manipulations and calculations:

.. code-block:: python

    from general_python.physics import density_matrix, entropy
    from general_python.algebra.utils import get_global_backend
    
    backend = get_global_backend()
    
    # Create a quantum state (example: |+> state)
    psi = backend.array([1, 1]) / backend.sqrt(2)
    
    # Calculate density matrix
    rho = backend.outer(psi, psi.conj())
    print("Density matrix shape:", rho.shape)
    
    # Calculate von Neumann entropy (if functions are available)
    try:
        s_vn = entropy.von_neumann_entropy(rho)
        print("von Neumann entropy:", s_vn)
    except AttributeError:
        print("Entropy calculation functions may need to be implemented")

**6. Data Handling and Visualization**

Manage data and create scientific plots:

.. code-block:: python

    from general_python.common import Directories
    from general_python.common.plot import Plotter
    import numpy as np
    
    # Directory management
    dir_handler = Directories("./data")
    dir_handler.create_directory("./data/results")
    
    # Create scientific plots
    plotter = Plotter()
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    
    plotter.plot(x, y, label="sin(x)")
    plotter.set_labels("x", "y", "Sine Function")
    plotter.show()

These examples demonstrate the library's flexibility and comprehensive functionality. The automatic backend detection ensures optimal performance whether running on CPU (NumPy) or accelerated hardware (JAX). For detailed API documentation, see the API Reference section.
