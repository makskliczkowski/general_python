import numpy as np
import scipy.sparse as sp

def create_spin_chain_hamiltonian(n_spins: int):
    r"""
    Create Heisenberg spin chain Hamiltonian (dense).
    H = \Sigma _i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + S^z_i S^z_{i+1})

    Returns a dense numpy array.
    """
    n_states = 2**n_spins
    # Safety check for memory
    if n_states > 16384: # n=14
         print(f"Warning: Allocating large dense matrix ({n_states}x{n_states})")

    H = np.zeros((n_states, n_states))

    # Build Hamiltonian
    for state in range(n_states):
        for site in range(n_spins - 1):
            # S^z_i S^z_{i+1}
            bit_i = (state >> site) & 1
            bit_ip1 = (state >> (site + 1)) & 1
            sz_i = 0.5 if bit_i == 0 else -0.5
            sz_ip1 = 0.5 if bit_ip1 == 0 else -0.5
            H[state, state] += sz_i * sz_ip1

            # S^x_i S^x_{i+1} + S^y_i S^y_{i+1} = 0.5 * (S+_i S-_{i+1} + S-_i S+_{i+1})
            # These flip both spins if they are antiparallel
            new_state = state ^ (1 << site) ^ (1 << (site + 1))
            H[new_state, state] += 0.5

    return H

def create_sparse_spin_chain_hamiltonian(n_spins: int):
    r"""
    Create Heisenberg spin chain Hamiltonian as a sparse matrix (CSR).
    H = \Sigma _i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + S^z_i S^z_{i+1})
    """
    n_states = 2**n_spins

    rows = []
    cols = []
    data = []

    # Vectorized construction
    # Use int64 to avoid overflow for large states if needed, though n_spins <= 20 fits in int32
    states = np.arange(n_states, dtype=np.int64)

    # Diagonal: Sum S^z_i S^z_{i+1}
    diag = np.zeros(n_states, dtype=np.float64)

    for i in range(n_spins - 1):
        # bit_i: (states >> i) & 1
        # bit_ip1: (states >> (i+1)) & 1
        # If bits equal (00 or 11), sz*sz = 0.5*0.5 = 0.25
        # If bits diff (01 or 10), sz*sz = 0.5*-0.5 = -0.25

        mask_same = ((states >> i) & 1) == ((states >> (i + 1)) & 1)
        term = np.where(mask_same, 0.25, -0.25)
        diag += term

        # Off-diagonal: 0.5 * (S+ S- + S- S+)
        # Valid only if bits are different (mask_same is False)
        # Flip both bits at i and i+1

        valid_flips = ~mask_same
        # Indices where flip is valid
        # We can just extract them
        src_states = states[valid_flips]

        flip_mask = (1 << i) | (1 << (i + 1))
        dst_states = src_states ^ flip_mask

        rows.append(dst_states)
        cols.append(src_states)
        data.append(np.full(len(src_states), 0.5))

    # Add diagonal
    rows.append(states)
    cols.append(states)
    data.append(diag)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)

    return sp.coo_matrix((data, (rows, cols)), shape=(n_states, n_states)).tocsr()

def create_convection_diffusion_matrix(n: int, c: float = 10.0):
    """
    Create a non-symmetric sparse matrix representing 1D convection-diffusion.
    -u'' + c u'
    Discretized on [0, 1] with finite differences.
    """
    h = 1.0 / (n + 1)

    # Coefficients
    # u_i term: 2/h^2
    # u_{i-1} term: -1/h^2 - c/2h
    # u_{i+1} term: -1/h^2 + c/2h

    diag_val = 2.0 / h**2
    off_m1_val = -1.0 / h**2 - c / (2*h)
    off_p1_val = -1.0 / h**2 + c / (2*h)

    diag = np.full(n, diag_val)
    off_diag_m1 = np.full(n-1, off_m1_val)
    off_diag_p1 = np.full(n-1, off_p1_val)

    data = [diag, off_diag_m1, off_diag_p1]
    offsets = [0, -1, 1]

    return sp.diags(data, offsets, format='csr')
