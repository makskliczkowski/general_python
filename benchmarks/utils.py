import numpy as np

def create_spin_chain_hamiltonian(n_spins: int):
    """
    Create Heisenberg spin chain Hamiltonian.
    H = \Sigma _i (\sigma ^x_i \sigma ^x_{i+1} + \sigma ^y_i \sigma ^y_{i+1} + \sigma ^z_i \sigma ^z_{i+1})

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
            # \sigma ^z_i \sigma ^z_{i+1}
            bit_i = (state >> site) & 1
            bit_ip1 = (state >> (site + 1)) & 1
            sz_i = 0.5 if bit_i == 0 else -0.5
            sz_ip1 = 0.5 if bit_ip1 == 0 else -0.5
            H[state, state] += sz_i * sz_ip1

            # \sigma ^x_i \sigma ^x_{i+1} + \sigma ^y_i \sigma ^y_{i+1}
            # These flip both spins
            new_state = state ^ (1 << site) ^ (1 << (site + 1))
            H[new_state, state] += 0.5

    return H
