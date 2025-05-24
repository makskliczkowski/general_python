Introduction
============

General Python Utilities is a comprehensive Python library offering a broad set of tools and convenience functions for scientific computing, with particular emphasis on quantum physics simulations and numerical methods. It consolidates various frequently used functionalities into a single, easy-to-use package with flexible backend support for both NumPy and JAX.

**Key Features:**

🧮 **Advanced Algebra & Linear Algebra**
   - Comprehensive linear algebra operations with automatic NumPy/JAX backend detection
   - Sparse matrix operations and specialized solvers (CG, MinRes-QLP, direct methods)
   - Eigenvalue/eigenvector computations with optimized routines
   - Preconditioners for iterative solvers
   - ODE solving utilities

🎲 **Mathematics & Random Number Generation**
   - High-quality pseudorandom number generators (e.g., Xoshiro256 algorithm)
   - Comprehensive statistical functions and data analysis tools
   - Mathematical utilities and special functions
   - Reproducible random sequences for scientific computing

🔗 **Lattice Structures**
   - Tools for creating and manipulating lattice geometries
   - Support for square, hexagonal, and honeycomb lattices
   - Efficient neighbor finding and lattice navigation algorithms
   - Visualization utilities for lattice structures

🧠 **Machine Learning Framework**
   - Neural network implementations with flexible JAX/NumPy backends
   - Training utilities, optimizers, and learning rate schedulers
   - Loss functions for various ML tasks
   - Integration with modern ML workflows

⚛️ **Quantum Physics Utilities**
   - Density matrix operations and manipulations
   - Quantum entropy calculations (von Neumann, Rényi)
   - Eigenstate analysis and quantum operator utilities
   - JAX-optimized quantum computations

🛠️ **Common Utilities**
   - File and directory management with advanced I/O
   - HDF5 data handling and serialization
   - Plotting and visualization tools with scientific styling
   - Comprehensive logging and debugging utilities
   - Binary operations and bit manipulation tools

**Performance & Flexibility:**
The library is designed with performance in mind, leveraging optimized libraries like NumPy, SciPy, and JAX. The automatic backend detection allows seamless switching between NumPy (CPU) and JAX (CPU/GPU/TPU) for maximum computational efficiency.

This library serves as both a standalone toolbox for scientific programming and a foundational component for larger physics simulation frameworks, particularly in quantum many-body systems and condensed matter physics research.
