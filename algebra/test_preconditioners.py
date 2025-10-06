'''
file:       general_python/algebra/test_preconditioners.py
author:     Maksymilian Kliczkowski
desc:       Unit tests for the Preconditioner classes and factory.
'''

import time
import pytest
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

# Get the code to test
from ..algebra.preconditioners import (Preconditioner,
                                                    IdentityPreconditioner,
                                                    JacobiPreconditioner,
                                                    CholeskyPreconditioner,
                                                    SSORPreconditioner,
                                                    IncompleteCholeskyPreconditioner,
                                                    PreconditionersType,
                                                    PreconditionersTypeSym,
                                                    PreconditionersTypeNoSym,
                                                    choose_precond,
                                                    Array,
                                                    JAX_AVAILABLE,
                                                    get_backend)
from ..algebra.solver import Solver, SolverResult

# Conditionally import JAX
if JAX_AVAILABLE:
    try:
        import jax
        import jax.numpy as jnp
        
        @jax.jit
        def _dummy_jax_solver_step(matvec, precond_apply, r, x):
            """
            A dummy step that uses matvec and precond_apply
            """
            
            # Apply preconditioner Mz = r -> z = M^-1 r
            z = precond_apply(r) 
            # Use z in some way, e.g., update direction p = z + beta*p
            # Use matvec, e.g., Ap = matvec(p)
            # Here, just return something based on z and matvec(x)
            return x + z + matvec(x)

    except ImportError:
        JAX_AVAILABLE          = False
        jax                     = None
        jnp                     = None
        _dummy_jax_solver_step  = None
else:
    jax                         = None
    jnp                         = np
    _dummy_jax_solver_step      = None

# Determine available backends for testing
_available_backends = ['numpy']
if JAX_AVAILABLE:
    _available_backends.append('jax')

# Tolerances for numerical comparisons
_TOL = 1e-8

# =====================================================================
#! Test Fixtures
# =====================================================================

@pytest.fixture(scope="module", params=_available_backends)
def fixture_backend(request):
    """
    Fixture to provide backend string ('numpy', 'jax')
    for the tests.
    This allows parameterized testing across different backends.
    
    Parameters:
    ----------
        request: pytest.FixtureRequest
            The request object for the fixture.
    Returns:
    -------
        str: 
            The backend string ('numpy', 'jax').
    """
    return request.param

@pytest.fixture(scope="module")
def fixture_matrix_data(fixture_backend):
    """
    Fixture to generate various test matrices and vectors for the specified backend.
    Returns a dictionary of matrices/vectors and the backend module.

    Args:
        fixture_backend (str): The backend string ('numpy', 'jax').

    Returns:
        tuple[dict, Any]:
            - Dictionary of test matrices/vectors ('A_spd', 'r_vec', etc.).
            - The backend module (e.g., numpy or jax.numpy).
    """
    bcknd            = fixture_backend
    backend_module, _ = get_backend(bcknd)
    np_dtype         = np.float64 # Use float64 for numpy tests for precision

    # --- Define NumPy Matrices First ---
    # Symmetric Positive Definite (SPD)
    a_spd_np        = np.array([[ 4.0,  1.0,  0.0],
                                [ 1.0,  5.0, -1.0],
                                [ 0.0, -1.0,  3.0]], dtype=np_dtype)
    # General Symmetric (may not be PD)
    a_sym_np        = np.array([[ 1.0,  2.0, -1.0],
                                [ 2.0, -1.0,  3.0],
                                [-1.0,  3.0,  0.0]], dtype=np_dtype)
    # Non-Symmetric
    a_nonsym_np     = np.array([[ 1.0,  2.0,  3.0],
                                [ 4.0,  5.0,  6.0],
                                [ 7.0,  8.0,  0.0]], dtype=np_dtype)
    # Gram factors (S)
    s_np            = np.array([[ 1.0,  2.0],
                                [ 3.0,  4.0],
                                [-1.0,  0.0]], dtype=np_dtype)  # Shape (3, 2)
    sp_np           = s_np.T.conj()                             # Shape (2, 3)

    # Vectors
    # Compatible with 3x3 matrices
    b_vec_np        = np.array([1.0, 2.0, 3.0], dtype=np_dtype)
    r_vec_np        = np.array([0.5, -0.1, 1.2], dtype=np_dtype)
    # Compatible with 2x2 system from Gram matrix
    r_vec_gram_np   = np.array([0.1, -0.2], dtype=np_dtype)

    # Sparse Matrix (for ILU/IC tests - always NumPy/SciPy based)
    a_sparse_spd_np = sps.csc_matrix(a_spd_np)
    s_sparse_np     = sps.csc_matrix(s_np)
    sp_sparse_np    = sps.csc_matrix(sp_np)

    # --- Convert to Backend ---
    # Use backend_module.asarray for conversion
    matrix_dict = {
        'A_spd'         : backend_module.asarray(a_spd_np),
        'A_sym'         : backend_module.asarray(a_sym_np),
        'A_nonsym'      : backend_module.asarray(a_nonsym_np),
        'S'             : backend_module.asarray(s_np),
        'Sp'            : backend_module.asarray(sp_np),
        'b_vec'         : backend_module.asarray(b_vec_np),
        'r_vec'         : backend_module.asarray(r_vec_np),
        'r_vec_gram'    : backend_module.asarray(r_vec_gram_np),
        # Keep sparse matrices as NumPy/SciPy objects
        'A_sparse_spd'  : a_sparse_spd_np,
        'S_sparse'      : s_sparse_np,
        'Sp_sparse'     : sp_sparse_np
    }
    return matrix_dict, backend_module

# =====================================================================
#! Test Class for Preconditioners
# =====================================================================

class TestPreconditioners:
    """
    Test suite for Preconditioner classes.
    """

    # -----------------------------------------------------------------
    #! Factory Function Tests
    # -----------------------------------------------------------------

    def test_choose_precond_factory(self, fixture_backend):
        """
        Tests the choose_precond factory function for correct instantiation.
        """
        bcknd = fixture_backend

        # Test with Enum
        p_jacobi = choose_precond(PreconditionersTypeSym.JACOBI, backend=bcknd)
        assert isinstance(p_jacobi, JacobiPreconditioner)
        assert p_jacobi.backend_str == bcknd

        # Test with string (case-insensitive implicit in factory)
        p_id = choose_precond('identity', backend=bcknd)
        assert isinstance(p_id, IdentityPreconditioner)
        assert p_id.backend_str == bcknd

        # Test passing an instance
        p_existing = SSORPreconditioner(backend=bcknd)
        p_chosen = choose_precond(p_existing, backend='numpy') # Backend kwarg should be ignored
        assert p_chosen is p_existing
        assert p_chosen.backend_str == bcknd # Original backend preserved

        # Test with kwargs propagation
        test_tol_small = 1e-12
        test_zero_replacement = 1e12
        p_jacobi_custom = choose_precond(
            'JACOBI',
            backend=bcknd,
            tol_small=test_tol_small,
            zero_replacement=test_zero_replacement
        )
        assert isinstance(p_jacobi_custom, JacobiPreconditioner)
        assert p_jacobi_custom.tol_small == test_tol_small
        assert p_jacobi_custom.zero_replacement == test_zero_replacement

        # Test invalid identifier
        with pytest.raises(ValueError):
            choose_precond('INVALID_PRECOND', backend=bcknd)
        with pytest.raises(TypeError):
            choose_precond(123.45, backend=bcknd)

    # -----------------------------------------------------------------
    #! Identity Preconditioner Tests
    # -----------------------------------------------------------------

    def test_identity_preconditioner(self, fixture_matrix_data, fixture_backend):
        """ 
        Tests the IdentityPreconditioner setup and apply.
        """
        
        matrix_dict, backend_module = fixture_matrix_data
        r_test_vec                  = matrix_dict['r_vec']
        bcknd                       = fixture_backend

        precond                     = IdentityPreconditioner(backend=bcknd)
        assert precond.type         == PreconditionersTypeSym.IDENTITY
        assert precond.stype        == PreconditionersType.SYMMETRIC

        # Test Setup
        t_start                     = time.perf_counter()
        # Set should be a no-op, doesn't store anything
        precond.set(matrix_dict['A_spd'], sigma=0.1)
        t_end                       = time.perf_counter()
        print(f"\n  [Identity-{bcknd}] Setup time: {t_end - t_start:.6f} s")
        assert precond.precomputed_data == {}

        # Test Apply (via __call__)
        t_start                     = time.perf_counter()
        r_preconditioned            = precond(r_test_vec)
        t_end                       = time.perf_counter()
        print(f"  [Identity-{bcknd}] Apply time (__call__): {t_end - t_start:.6f} s")
        np.testing.assert_allclose(
            backend_module.to_numpy(r_preconditioned),
            backend_module.to_numpy(r_test_vec),
            rtol=_TOL, atol=_TOL
        )

        # Test Apply (via get_apply)
        precond_apply_func          = precond.get_apply()
        t_start                     = time.perf_counter()
        r_preconditioned_instance   = precond_apply_func(r_test_vec)
        t_end                       = time.perf_counter()
        print(f"  [Identity-{bcknd}] Apply time (get_apply): {t_end - t_start:.6f} s")
        assert np.allclose(
            backend_module.to_numpy(r_preconditioned_instance),
            backend_module.to_numpy(r_test_vec),
            rtol=_TOL, atol=_TOL
        )

        # Test Static Apply
        # Note: Static apply doesn't use instance state or timing
        r_preconditioned_static = IdentityPreconditioner.apply(r=r_test_vec, backend_mod=backend_module)
        np.testing.assert_allclose(
            backend_module.to_numpy(r_preconditioned_static),
            backend_module.to_numpy(r_test_vec),
            rtol=_TOL, atol=_TOL
        )

    # -----------------------------------------------------------------
    #! Jacobi Preconditioner Tests
    # -----------------------------------------------------------------

    def test_jacobi_preconditioner_standard(self, fixture_matrix_data, fixture_backend):
        """ 
        Tests JacobiPreconditioner with standard A matrix setup.
        """
        matrix_dict, backend_module = fixture_matrix_data
        a_spd                       = matrix_dict['A_spd']
        r_test_vec                  = matrix_dict['r_vec']
        sigma                       = 0.1
        bcknd                       = fixture_backend

        precond                     = JacobiPreconditioner(backend=bcknd, is_positive_semidefinite=True)
        assert precond.type        == PreconditionersTypeSym.JACOBI
        assert precond.stype       == PreconditionersType.SYMMETRIC

        # Test Setup
        t_start                     = time.perf_counter()
        precond.set(a_spd, sigma=sigma)
        t_end                       = time.perf_counter()
        print(f"\n  [Jacobi(Std)-{bcknd}] Setup time: {t_end - t_start:.6f} s")

        # Check precomputed data
        precomputed_data            = precond.precomputed_data
        assert 'inv_diag' in precomputed_data
        # Expected calculation uses safe division logic from implementation for comparison
        diag_a_spd                  = backend_module.diag(a_spd)
        inv_diag_expected           = precond._compute_inv_diag(diag_a_spd, sigma)
        np.testing.assert_allclose(
            backend_module.to_numpy(precomputed_data['inv_diag']),
            backend_module.to_numpy(inv_diag_expected),
            rtol=_TOL, atol=_TOL
        )

        # Test Apply (via get_apply)
        precond_apply_func          = precond.get_apply()
        # First call might include JIT compilation time for JAX
        _                           = precond_apply_func(r_test_vec)
        t_start                     = time.perf_counter()
        r_preconditioned            = precond_apply_func(r_test_vec)
        t_end                       = time.perf_counter()
        print(f"  [Jacobi(Std)-{bcknd}] Apply time (get_apply): {t_end - t_start:.6f} s")

        # Verify result
        r_expected                  = inv_diag_expected * r_test_vec
        np.testing.assert_allclose(
            backend_module.to_numpy(r_preconditioned),
            backend_module.to_numpy(r_expected),
            rtol=_TOL, atol=_TOL
        )

        # Test Static Apply
        r_preconditioned_static     = JacobiPreconditioner.apply(
            r           = r_test_vec,
            backend_mod = backend_module,
            **precomputed_data
        )
        np.testing.assert_allclose(
            backend_module.to_numpy(r_preconditioned_static),
            backend_module.to_numpy(r_expected),
            rtol=_TOL, atol=_TOL
        )

    def test_jacobi_preconditioner_gram(self, fixture_matrix_data, fixture_backend):
        """ Tests JacobiPreconditioner with Gram matrix factors S, Sp. """
        matrix_dict, backend_module = fixture_matrix_data
        s_mat                       = matrix_dict['S']
        sp_mat                      = matrix_dict['Sp']
        r_vec_gram                  = matrix_dict['r_vec_gram'] # Use the correct dim vector
        sigma                       = 0.1
        n_norm                      = float(s_mat.shape[0]) # Normalization factor N
        bcknd                       = fixture_backend
        precond                     = JacobiPreconditioner(backend=bcknd, is_gram=True)

        # Test Setup
        t_start                     = time.perf_counter()
        precond.set(s_mat, sigma=sigma, ap=sp_mat) # Provide Sp via ap
        t_end                       = time.perf_counter()
        print(f"\n  [Jacobi(Gram)-{bcknd}] Setup time: {t_end - t_start:.6f} s")

        # Calculate expected diagonal of A = Sp @ S / N for verification
        diag_a_gram_expected        = backend_module.einsum('ij,ji->i', sp_mat, s_mat) / n_norm
        inv_diag_expected           = precond._compute_inv_diag(diag_a_gram_expected, sigma)

        # Check precomputed data
        precomputed_data            = precond.precomputed_data
        assert 'inv_diag' in precomputed_data
        np.testing.assert_allclose(
            backend_module.to_numpy(precomputed_data['inv_diag']),
            backend_module.to_numpy(inv_diag_expected),
            rtol=_TOL, atol=_TOL
        )

        # Test Apply (via get_apply)
        precond_apply_func          = precond.get_apply()
        _                           = precond_apply_func(r_vec_gram) # Warm-up JIT if applicable
        t_start                     = time.perf_counter()
        r_preconditioned            = precond_apply_func(r_vec_gram)
        t_end                       = time.perf_counter()
        print(f"  [Jacobi(Gram)-{bcknd}] Apply time (get_apply): {t_end - t_start:.6f} s")

        # Verify result
        r_expected                  = inv_diag_expected * r_vec_gram
        np.testing.assert_allclose(
            backend_module.to_numpy(r_preconditioned),
            backend_module.to_numpy(r_expected),
            rtol=_TOL, atol=_TOL
        )

    def test_jacobi_preconditioner_zero_diag(self, fixture_backend):
        """ Tests JacobiPreconditioner handling of zero/small diagonal elements. """
        bcknd                       = fixture_backend
        backend_module, _           = get_backend(bcknd)
        np_dtype                    = np.float64
        # Matrix with a zero on diagonal
        a_zero_diag                 = backend_module.asarray(np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np_dtype))
        r_test_vec                  = backend_module.asarray(np.array([1.0, 1.0], dtype=np_dtype))
        test_tol_small              = 1e-9
        test_zero_repl              = 1e12 # Renamed variable

        # Use non-default tolerances
        precond = JacobiPreconditioner(
            backend         =   bcknd,
            tol_small       =   test_tol_small,
            zero_replacement=   test_zero_repl
        )

        # Test without sigma
        precond.set(a_zero_diag, sigma=0.0)
        inv_diag                    = precond.precomputed_data['inv_diag']
        # Expected: 1/(1), 1/(0 handled) -> 0
        inv_diag_expected           = backend_module.asarray([1.0 / 1.0, 0.0])
        np.testing.assert_allclose(
            backend_module.to_numpy(inv_diag),
            backend_module.to_numpy(inv_diag_expected),
            rtol=_TOL, atol=_TOL
        )
        precond_apply_func          = precond.get_apply()
        r_preconditioned            = precond_apply_func(r_test_vec)
        r_expected                  = inv_diag_expected * r_test_vec
        np.testing.assert_allclose(
            backend_module.to_numpy(r_preconditioned),
            backend_module.to_numpy(r_expected),
            rtol=_TOL, atol=_TOL
        )

        # Test with sigma that still leaves one element small
        sigma_small                 = 1e-10 # Smaller than tol_small
        precond.set(a_zero_diag, sigma=sigma_small)
        inv_diag                    = precond.precomputed_data['inv_diag']
        # Expected: 1/(1+sigma), 1/(0+sigma treated as small) -> 0
        inv_diag_expected           = backend_module.asarray([1.0 / (1.0 + sigma_small), 0.0])
        np.testing.assert_allclose(
            backend_module.to_numpy(inv_diag),
            backend_module.to_numpy(inv_diag_expected),
            rtol=_TOL, atol=_TOL
        )
        precond_apply_func          = precond.get_apply()
        r_preconditioned            = precond_apply_func(r_test_vec)
        r_expected                  = inv_diag_expected * r_test_vec
        np.testing.assert_allclose(
            backend_module.to_numpy(r_preconditioned),
            backend_module.to_numpy(r_expected),
            rtol=_TOL, atol=_TOL
        )

    # -----------------------------------------------------------------
    #! Backend Switching and JIT Interaction Tests
    # -----------------------------------------------------------------

    def test_backend_switching(self, fixture_matrix_data):
        """ 
        Tests switching backend after instantiation and re-setting.
        """
        if not JAX_AVAILABLE:
            pytest.skip("Backend switching test requires JAX.")

        # Need data for both backends
        mats_np, be_np      = fixture_matrix_data('numpy')
        mats_jax, be_jax    = fixture_matrix_data('jax')

        a_np                = mats_np['A_spd']
        r_np                = mats_np['r_vec']
        a_jax               = mats_jax['A_spd']
        r_jax               = mats_jax['r_vec']
        sigma               = 0.1

        # Start with NumPy
        precond             = JacobiPreconditioner(backend='numpy')
        precond.set(a_np, sigma=sigma)
        precond_apply_np    = precond.get_apply()
        r_precond_np        = precond_apply_np(r_np)
        assert isinstance(r_precond_np, np.ndarray)
        print("\n  [BackendSwitch] NumPy result obtained.")

        # --- Switch to JAX ---
        precond.reset_backend('jax')
        assert precond.backend_str == 'jax'
        assert precond.precomputed_data is None     # Reset invalidates data
        
        # Re-set with JAX matrix
        precond.set(a_jax, sigma=sigma)             # Uses JAX arrays now
        precond_apply_jax    = precond.get_apply()  # Get potentially JITted function
        # Run apply (might compile here)
        r_precond_jax        = precond_apply_jax(r_jax)
        # Run again for timing (post-compile)
        t_start              = time.perf_counter()
        r_precond_jax        = precond_apply_jax(r_jax)
        t_end               = time.perf_counter()
        print(f"  [BackendSwitch] JAX apply time (get_apply): {t_end-t_start:.6f}s")
        assert isinstance(r_precond_jax, jnp.ndarray) # Check JAX array type

        # Verify results are numerically close
        np.testing.assert_allclose(
            r_precond_np,
            be_jax.to_numpy(r_precond_jax),
            rtol=_TOL, atol=_TOL
        )
        print("  [BackendSwitch] NumPy and JAX results match.")

        # --- Switch back to NumPy ---
        precond.reset_backend('numpy')
        assert precond.backend_str == 'numpy'
        assert precond.precomputed_data is None
        precond.set(a_np, sigma=sigma)              # Re-set with NumPy matrix
        precond_apply_np_again  = precond.get_apply()
        r_precond_np_again      = precond_apply_np_again(r_np)
        assert isinstance(r_precond_np_again, np.ndarray)
        np.testing.assert_allclose(r_precond_np, r_precond_np_again, rtol=_TOL, atol=_TOL)
        print("  [BackendSwitch] Switched back to NumPy, result matches.")

    # -----------------------------------------------------------------
    #! JIT Interaction Tests
    # -----------------------------------------------------------------

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="Requires JAX for JIT testing")
    def test_preconditioner_with_jit_solver(self, fixture_matrix_data):
        """
        Tests using a JAX-JITted preconditioner apply func within a JITted step.
        """
        bcknd                       = 'jax'
        matrix_dict, backend_module = fixture_matrix_data(bcknd) # Get JAX data/module
        a_jax                       = matrix_dict['A_spd']
        r_jax                       = matrix_dict['r_vec']
        x_jax                       = backend_module.zeros_like(r_jax) # Dummy x vector
        sigma                       = 0.1

        # Test with Jacobi
        precond_jac                 = JacobiPreconditioner(backend='jax')
        precond_jac.set(a_jax, sigma=sigma)
        # Get the potentially JIT-compiled apply function from the instance
        precond_apply_jac           = precond_jac.get_apply()
        print(f"\n  [JIT Interaction] Jacobi apply func type: {type(precond_apply_jac)}")
        # Check it seems like a JITted function (heuristic)
        assert hasattr(precond_apply_jac, "lower") or "CompiledFunction" in str(type(precond_apply_jac))

        # Create a dummy matvec function (JAX compatible)
        # Define inside or ensure closure captures a_jax correctly if defined outside
        @jax.jit
        def matvec_jacobi_jax(v):
            return backend_module.dot(a_jax, v)

        # Call the dummy JITted solver step, passing the preconditioner's apply function
        # The _dummy_jax_solver_step function itself is JITted
        print("  [JIT Interaction] Calling JITted solver step with Jacobi...")
        # First call will trigger compilation of _dummy_jax_solver_step and potentially precond_apply_jac
        result_jac_jax              = _dummy_jax_solver_step(matvec_jacobi_jax, precond_apply_jac, r_jax, x_jax)
        # Second call should be faster
        t_start                     = time.perf_counter()
        result_jac_jax              = _dummy_jax_solver_step(matvec_jacobi_jax, precond_apply_jac, r_jax, x_jax)
        t_end                       = time.perf_counter()
        print(f"  [JIT Interaction] JITted step execution time (Jacobi): {t_end - t_start:.6f} s")

        # Check the result type and shape
        assert isinstance(result_jac_jax, jnp.ndarray)
        assert result_jac_jax.shape == r_jax.shape

        # Calculate expected result manually
        inv_diag_expected           = 1.0 / (backend_module.diag(a_jax) + sigma)
        z_expected                  = inv_diag_expected * r_jax
        ax_expected                 = backend_module.dot(a_jax, x_jax)
        result_expected             = x_jax + z_expected + ax_expected
        np.testing.assert_allclose(
            backend_module.to_numpy(result_jac_jax),
            backend_module.to_numpy(result_expected),
            rtol=_TOL, atol=_TOL
        )
        print("  [JIT Interaction] Jacobi JITted step result verified.")

    # -----------------------------------------------------------------
    #! General Error Handling and Edge Cases
    # -----------------------------------------------------------------

    def test_apply_before_set(self, fixture_backend):
        """ Tests calling apply before set raises appropriate errors. """
        bcknd   = fixture_backend
        precond = JacobiPreconditioner(backend=bcknd)
        # Need a dummy vector compatible with the backend
        be, _   = get_backend(bcknd)
        r_dummy = be.asarray(np.array([1.0, 2.0]))

        # Check internal data retrieval before set
        with pytest.raises(RuntimeError, match="Preconditioner data not available"):
            _ = precond.precomputed_data # Should be None or raise

        # Check getting the apply function before set
        with pytest.raises(RuntimeError, match="could not be initialized"):
            _ = precond.get_apply() # Should fail

        # Check calling the instance (__call__) before set
        with pytest.raises(RuntimeError, match="could not be initialized"):
            _ = precond(r_dummy)

    def test_shape_mismatch_apply(self, fixture_matrix_data, fixture_backend):
        """ 
        Tests errors raised during apply/call due to shape mismatches.
        """
        matrix_dict, backend_module = fixture_matrix_data
        a_spd                       = matrix_dict['A_spd'] # 3x3
        # Create vector of wrong size
        r_wrong_size                = backend_module.asarray(np.array([1.0, 2.0])) # size 2
        bcknd                       = fixture_backend

        # Jacobi
        precond_jac                 = JacobiPreconditioner(backend=bcknd)
        precond_jac.set(a_spd)
        precond_jac_apply           = precond_jac.get_apply()
        # Error expected inside the apply function
        with pytest.raises(ValueError, match=r".*Shape mismatch.*Jacobi apply.*|.*Dimension mismatch.*Jacobi apply.*"):
            _ = precond_jac_apply(r_wrong_size)

# =====================================================================
#! End of Test Class
# =====================================================================