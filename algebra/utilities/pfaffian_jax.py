from general_python.algebra.utils import JAX_AVAILABLE, Array

#! jax
if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax import jit
else:
    jax = None
    jnp = None
    jit = None
    
################################################################################################################

_ZERO_TOL       = 1e-11

if JAX_AVAILABLE:
    @jit
    def _check_skew_symmetric_jax(A: jnp.array, tol=_ZERO_TOL):
        """
        Checks if a matrix A is skew-symmetric within a tolerance using JAX.
        A matrix is skew-symmetric if A^T = -A.
        Parameters:
            A (Array):
                The matrix to check.
            tol (float):
                The tolerance for checking skew-symmetry.
        Returns:
            bool: True if A is skew-symmetric, False otherwise.
        """
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Input must be a square matrix.")
        return jnp.allclose(A, -A.T, atol=tol)
else:
    def _check_skew_symmetric_jax(A: Array, tol=1e-9):
        """
        Placeholder for JAX skew-symmetric check.
        This function is not used if JAX is not available.
        """
        raise NotImplementedError("JAX is not available. Skew-symmetric check cannot be performed.")

###############################################################################################################

if JAX_AVAILABLE:
    
    class Pfaffian:
        """
        Provides Pfaffian and related calculations using JAX JIT.
        """

        ########################################################################
        #! Recursive Pfaffian (JAX)
        ########################################################################
        
        @classmethod
        def _pfaffian_recursive_jax(cls, A, N):
            """
            Calculates the Pfaffian using the recursive definition (JAX version).
            WARNING: O(N!) complexity. JIT compilation is problematic.
            """
            if N == 0:
                return jnp.array(1.0, dtype=A.dtype)
            if N % 2 != 0:
                return jnp.array(0.0, dtype=A.dtype)

            if N == 2:
                return A[0, 1]

            pf_sum          = jnp.array(0.0, dtype=A.dtype)
            fixed_row_col   = 0
            all_indices     = jnp.arange(N, dtype=jnp.int32)

            # Python loop: prevents JIT unless N is static
            for j in range(1, N):
                sign_j  = jnp.array(1.0 if (j - 1) % 2 == 0 else -1.0, dtype=A.dtype)

                N_minor = N - 2
                if N_minor >= 0:
                    # Create indices for minor (dynamic, hard for JIT)
                    mask            = jnp.ones(N, dtype=jnp.bool_)
                    mask            = mask.at[fixed_row_col].set(False)
                    mask            = mask.at[j].set(False)
                    minor_indices   = all_indices[mask]

                    # Create minor matrix using advanced indexing
                    A_minor         = A[jnp.ix_(minor_indices, minor_indices)]

                    # Recursive call
                    pf_minor        = cls._pfaffian_recursive_jax(A_minor, N_minor)
                    pf_sum          += sign_j * A[fixed_row_col, j] * pf_minor
            return pf_sum

        ########################################################################
        #! Hessenberg Pfaffian (JAX)
        ########################################################################
        
        @staticmethod
        @jit
        def _pfaffian_hessenberg_jax(A, N):
            """
            Calculates the Pfaffian using the Hessenberg form (JAX JIT).
            WARNING:
                Assumes det(Q)=1 for the orthogonal transformation Q,
                as jnp.linalg.hessenberg does not return Q or det(Q).
                Result might be incorrect by a factor of -1.
            """
            # N=0 and N%2!=0 handled by wrapper
            H           = jnp.linalg.hessenberg(A)

            # Pf(H) = H[0,1] * H[2,3] * ...
            super_diag  = jnp.diag(H, k=1)
            indices     = jnp.arange(0, N - 1, 2, dtype=jnp.int32)
            pf_H        = jnp.prod(jnp.take(super_diag, indices))

            return pf_H

        ########################################################################
        #! Schur Pfaffian (JAX)
        ########################################################################
        
        @staticmethod
        def _pfaffian_schur_jax(A, N):
            """
            Pfaffian via Schur decomposition - Not available in JAX.
            """
            raise NotImplementedError("jax.numpy.linalg does not provide a Schur decomposition.")

        ########################################################################
        #! Parlett-Reid Pfaffian (JAX)
        ########################################################################

        @staticmethod
        @jit
        def _pfaffian_parlett_reid_jax(A_in, N):
            """
            Internal JAX JIT: 
            Pfaffian via Parlett-Reid algorithm.
            Handles real and complex types using functional updates.
            """

            #! Helper for swapping rows/cols functionally
            def swap_rows_cols(mat, r1, r2):
                # Assumes r1 != r2
                row1    = mat[r1, :]
                row2    = mat[r2, :]
                mat     = mat.at[r1, :].set(row2)
                mat     = mat.at[r2, :].set(row1)
                # Swap cols using updated matrix
                col1    = mat[:, r1]
                col2    = mat[:, r2]
                mat     = mat.at[:, r1].set(col2)
                mat     = mat.at[:, r2].set(col1)
                return mat

            #! Loop body for lax.fori_loop
            def loop_body(k_idx, state):
                A, pfaffian_val = state
                k               = k_idx * 2

                #! 1. Pivoting
                # Slice: A[k+1:N, k] using dynamic slicing
                col_slice_size  = N - (k + 1)
                # Avoid slicing if size is zero or less
                col_slice       = lax.cond( col_slice_size > 0,
                                            lambda M: lax.dynamic_slice_in_dim(M[:, k], k + 1, col_slice_size),
                                            lambda M: jnp.empty(0, dtype=M.dtype), # Return empty array
                                            A)

                # Argmax (handle empty slice: argmax returns 0, kp=k+1)
                # Returns 0 if col_slice is empty or all zeros
                kp_offset       = jnp.argmax(jnp.abs(col_slice))
                kp              = (k + 1) + kp_offset
                condition       = jnp.logical_and(kp != k + 1, col_slice_size > 0)
                # Perform swap conditionally
                A_swapped = lax.cond(
                    condition,
                    lambda x: swap_rows_cols(x, k + 1, kp), # true_fun: swap
                    lambda x: x,                            # false_fun: identity
                    A
                )
                # Perform sign flip conditionally
                pf_sign = lax.cond(
                    condition,
                    lambda _: -1.0, # true_fun: multiplier is -1
                    lambda _: 1.0,  # false_fun: multiplier is 1
                    None
                )

                #! 2. Check pivot
                pivot_val_swapped   = A_swapped[k + 1, k]
                is_pivot_zero       = jnp.abs(pivot_val_swapped) < _ZERO_TOL

                #! 3. Pfaffian factor for this step
                # Use jnp.where to handle potential zero pivot
                pf_factor = jnp.where(is_pivot_zero,
                                    jnp.array(0.0, dtype=A.dtype),
                                    A_swapped[k, k + 1])

                #! 4. Calculate update matrix (only if needed)
                # Helper to calculate the update matrix term = outer(tau, col) - outer(col, tau).T
                def calc_update_matrix(A_update):
                    slice_start     = k + 2
                    slice_size      = N - slice_start
                    # Calculate tau and col safely inside the 'true' branch
                    tau             = lax.dynamic_slice(A_update[k, :], (slice_start,), (slice_size,)) / A_update[k, k + 1]
                    col             = lax.dynamic_slice(A_update[:, k + 1], (slice_start,), (slice_size,))

                    # A += row_times_col - col_times_row
                    # row_times_col = outer(tau, col)
                    # col_times_row = outer(col, tau)
                    col_times_row = jnp.outer(col, tau)
                    row_times_col = jnp.outer(tau, col)

                    return row_times_col - col_times_row

                # Condition to calculate actual update
                should_calculate_update = jnp.logical_and(jnp.logical_not(is_pivot_zero), k + 2 < N)
                # Placeholder zero matrix
                update_shape            = (N - (k + 2), N - (k + 2))
                zero_update             = jnp.zeros(update_shape, dtype=A.dtype)

                # Calculate update conditionally
                update = lax.cond(
                    should_calculate_update,
                    calc_update_matrix,       # True: calculate
                    lambda A_in: zero_update, # False: return zeros
                    A_swapped                 # Argument for the functions
                )

                #! 5. Update A
                # Apply update using dynamic_update_slice, only if needed
                A_next = lax.cond(
                    should_calculate_update,
                    lambda op: lax.dynamic_update_slice(op[0], op[1], (k + 2, k + 2)), # op = (A_swapped, update)
                    lambda op: op[0], # op = (A_swapped, update), return A_swapped
                    (A_swapped, update)
                )

                #! 6. Update pfaffian value
                pfaffian_next           = pfaffian_val * pf_sign * pf_factor
                return (A_next, pfaffian_next)
            
            #! End of loop_body

            # Initial state for the loop
            init_val = (A_in.copy(), jnp.array(1.0, dtype=A_in.dtype))

            #! Run the loop over pairs of indices (0, 2, 4, ...)
            final_A, final_pfaffian = lax.fori_loop(0, N // 2, loop_body, init_val)

            return final_pfaffian

        ########################################################################
        #! Cayley's Formula (JAX)
        ########################################################################
        
        @staticmethod
        @jit
        def _cayleys_formula_jax(_pffA, _Ainv_row, _updRow):
            """
            Internal JAX JIT implementation for Cayley's identity.
            P'(A) = -P(A) * dot(Ainv_row, updRow). Uses log space.
            """
            # Calculate dot product in log space
            dot_product = jnp.dot(_Ainv_row, _updRow)
            
            # Log-space version (assumes positivity):
            log_p   = jnp.log(_pffA)
            log_dot = jnp.log(dot_product) # Fails if dot_product <= 0
            return -jnp.exp(log_p + log_dot)
        
            # Add small epsilon to prevent log(0) or log(<0) if dot_product is non-positive
            # Note: This log-space approach assumes pfaffian and dot product are positive.
            # If they can be negative, the direct formula is safer.
            # Let's use the direct formula as it's more general.

            # Direct calculation:
            dot_product = jnp.dot(_Ainv_row, _updRow)
            return -_pffA * dot_product

        ########################################################################
        #! Sherman-Morrison Skew (JAX)
        ########################################################################
        
        @staticmethod
        @jit
        def _scherman_morrison_skew_jax(Ainv, updIdx, updRow):
            """
            Internal JAX JIT implementation for Sherman-Morrison update
            for skew-symmetric matrices.
            """
            N       = Ainv.shape[0]

            # Ensure updRow is treated as column for mat-vec product
            dots    = jnp.dot(Ainv, updRow)

            # Precompute inverse of the critical dot product with clipping
            dot_k   = dots[updIdx]
            
            # Clipping avoids NaN/Inf from exact zero, but might mask issues.
            dotProductInv = 1.0 / jnp.clip(dot_k, a_min=_ZERO_TOL, a_max=1e10)

            # Define the body for the nested loops using lax.fori_loop
            def body_j(j, state_i):
                i, current_row_vals     = state_i
                d_i_alpha               = jnp.where(i == updIdx, 1.0, 0.0)
                d_j_alpha               = jnp.where(j == updIdx, 1.0, 0.0)
                Ainv_k_i                = Ainv[updIdx, i]
                Ainv_k_j                = Ainv[updIdx, j]

                update_term             = dotProductInv * ( (d_i_alpha - dots[i]) * Ainv_k_j +
                                                            (dots[j] - d_j_alpha) * Ainv_k_i)
                new_val                 = current_row_vals[j] + update_term

                # Apply sign flip using jnp.where (Warning: reason unclear)
                sign_flip_condition     = jnp.logical_or(d_i_alpha > 0.5, d_j_alpha > 0.5)
                final_val               = jnp.where(sign_flip_condition, -new_val, new_val)

                # Return updated row values (will be used by .at[j].set outside loop)
                return current_row_vals.at[j].set(final_val)


            def body_i(i, current_out):
                # Inner loop updates row i
                # Initial row state for inner loop is current_out[i, :]
                init_row_state  = (i, current_out[i, :])
                # The body_j function updates the row values element by element
                # Fori_loop over j, updating the second element of init_row_state
                updated_row     = lax.fori_loop(0, N, body_j, init_row_state)[1] # Get updated row

                # Update the i-th row of the output matrix
                return current_out.at[i].set(updated_row)

            # Run the outer loop, starting with a copy (JAX handles copy implicitly)
            init_out_state  = Ainv
            final_out       = lax.fori_loop(0, N, body_i, init_out_state)

            return final_out

else: # JAX not available
    class PfaffianJAX:
        """Placeholder class if JAX is not installed."""
        @staticmethod
        def _pfaffian_recursive_jax(*args, **kwargs):
            raise ImportError("JAX not installed.")
        @staticmethod
        def _pfaffian_hessenberg_jax(*args, **kwargs):
            raise ImportError("JAX not installed.")
        @staticmethod
        def _pfaffian_schur_jax(*args, **kwargs):
            raise ImportError("JAX not installed.")
        @staticmethod
        def _pfaffian_parlett_reid_jax(*args, **kwargs):
            raise ImportError("JAX not installed.")
        @staticmethod
        def _cayleys_formula_jax(*args, **kwargs):
            raise ImportError("JAX not installed.")
        @staticmethod
        def _scherman_morrison_skew_jax(*args, **kwargs):
            raise ImportError("JAX not installed.")
        
################################################################################################################