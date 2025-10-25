# Physics Module: Structure and API

This document describes the structure and main features of the general_python/physics module. For mathematical background, see PHYSICS_MATH.md. For code examples, see PHYSICS_EXAMPLES.md.

## Directory Structure

general_python/physics/
    statistical.py
    thermal.py
    spectral/
        dos.py
        greens.py
        spectral_function.py
    response/
        structure_factor.py
        susceptibility.py
    density_matrix.py
    entropy.py
    operators.py
    ...

## Submodules and Capabilities

- statistical.py            : Moving averages, windowing, local density of states, binning
- thermal.py                : Partition function, thermal averages, susceptibilities, temperature scans
- spectral/                 : Density of states, Green's functions, spectral functions, Fourier transforms
- response/                 : Structure factors, susceptibilities, sum rules
- density_matrix.py         : Density matrices
- entropy.py, operators.py  : entropy measures, operator utilities

See PHYSICS_MATH.md for mathematical background and PHYSICS_EXAMPLES.md for usage examples.

## Copyright

Copyright (c) 2025 Maksymilian Kliczkowski. All rights reserved.
