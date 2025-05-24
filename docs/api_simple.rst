API Reference
=============

This section provides documentation for all modules and functions available in the General Python Utilities package.

Package Overview
----------------

The **General Python Utilities** package provides a comprehensive collection of mathematical, scientific, and machine learning tools organized into six main modules:

- **algebra**: Linear algebra operations and matrix utilities
- **common**: Common utilities, file handling, and helper functions  
- **lattices**: Lattice structures for physics simulations
- **maths**: Advanced mathematical functions and statistical tools
- **ml**: Machine learning utilities and neural networks
- **physics**: Quantum mechanics and physics simulation tools

Algebra Module
--------------

**Purpose**: Linear algebra utilities and mathematical operations.

**Key Features**:
- Matrix operations and transformations
- Linear algebra solvers
- Eigenvalue and eigenvector computations
- Vector space operations

**Main Classes and Functions**:

- ``Array``: Enhanced array operations with additional mathematical functionality
- ``linalg``: Linear algebra operations including eigenvalue decomposition
- ``solver``: Numerical solvers for linear systems
- ``utils``: Utility functions for array manipulation

Common Module  
-------------

**Purpose**: Common utilities and helper functions used across the package.

**Key Features**:
- File and directory management
- Data handling and I/O operations
- Logging and debugging utilities
- Plotting and visualization helpers

**Main Classes and Functions**:

- ``Directories``: Directory management and file system operations
- ``DataHandler``: Data loading, saving, and format conversion
- ``Plot``: Plotting utilities and visualization tools
- ``flog``: Logging and debugging functions

Lattices Module
---------------

**Purpose**: Lattice structures and operations for physics simulations.

**Key Features**:
- Various lattice geometries (square, hexagonal, triangular)
- Lattice site management and neighbor calculations
- Boundary condition handling
- Lattice-based physics simulations

**Main Classes and Functions**:

- ``Lattice``: Base lattice class with common functionality
- ``SquareLattice``: Square lattice implementation
- ``HexagonalLattice``: Hexagonal lattice structure
- ``TriangularLattice``: Triangular lattice geometry

Mathematics Module
------------------

**Purpose**: Advanced mathematical functions and numerical methods.

**Key Features**:
- Statistical analysis and distributions
- Numerical integration and differentiation
- Special functions and mathematical utilities
- Data analysis tools

**Main Classes and Functions**:

- ``Statistics``: Statistical analysis including histograms, CDF, and data processing
- ``Fraction``: Fraction operations and rational number arithmetic
- ``MathUtils``: General mathematical utilities and special functions
- ``Integration``: Numerical integration methods

Machine Learning Module
------------------------

**Purpose**: Machine learning utilities and neural network implementations.

**Key Features**:
- TensorFlow/Keras integration
- Custom neural network architectures
- Data preprocessing utilities
- Model training and evaluation tools

**Main Classes and Functions**:

- ``Networks``: Neural network implementations and architectures
- ``GeneralNet``: Flexible neural network base class
- ``CallableNet``: Callable neural network wrapper
- ``keras``: TensorFlow/Keras integration utilities

Physics Module
--------------

**Purpose**: Physics simulations and quantum mechanics utilities.

**Key Features**:
- Quantum state operations
- Density matrix calculations
- Entropy and entanglement measures
- Angular momentum operations

**Main Classes and Functions**:

- ``DensityMatrix``: Density matrix operations and quantum state analysis
- ``Entropy``: Entropy calculations and entanglement measures
- ``AngularMomentum``: Angular momentum operators and calculations
- ``QuantumStates``: Quantum state manipulation utilities

Usage Examples
--------------

Here are some basic usage examples for each module:

**Algebra Operations**::

    # Matrix operations
    from algebra.linalg import eigenvalues, eigenvectors
    eigenvals = eigenvalues(matrix)
    
**Statistical Analysis**::

    # Data analysis
    from maths.statistics import Statistics
    stats = Statistics(data)
    histogram = stats.get_histogram(bins=50)
    
**Lattice Creation**::

    # Create a square lattice
    from lattices.square import SquareLattice
    lattice = SquareLattice(dim=2, lx=10, ly=10, lz=1, bc='periodic')
    
**Machine Learning**::

    # Neural network setup
    from ml.networks import GeneralNet
    model = GeneralNet(input_size=784, hidden_sizes=[256, 128], output_size=10)
    
**Physics Calculations**::

    # Density matrix analysis
    from physics.density_matrix import DensityMatrix
    dm = DensityMatrix(quantum_state)
    entropy = dm.von_neumann_entropy()

For more detailed examples and advanced usage, see the :doc:`usage` section.
