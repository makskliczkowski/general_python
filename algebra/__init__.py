"""
A module for algebraic operations and utilities.
This module provides various linear algebra functions,
preconditioners, and solvers.
It supports both dense and sparse matrix computations and leverages different 
backends like NumPy and JAX to offer flexibility in performance and execution environments.

Key functionalities provided include:
    - Change-of-basis transformations for vectors and matrices.
    - Outer and Kronecker product computations.
    - Creation and handling of ket-bra operations.
    - Integration with preconditioners for iterative solvers.
    - Testing facilities for algebraic operations and linear solvers.

@Author: Maksymilian Kliczkowski
@Email: maksymilian.kliczkowski@pwr.edu.pl
@Date: 2025-02-01
@Version: 1.0
"""

import sys
import importlib
import time

# -----------------------------------------------------------------------------------------------

from general_python.algebra import linalg
from general_python.algebra import preconditioners
from general_python.algebra import solver
from general_python.algebra.solvers import SolverType, generate_test_mat_vec, choose_solver
from general_python.algebra.preconditioners import choose_precond, Preconditioner

from general_python.common.plot import MatrixPrinter
from general_python.common.flog import get_global_logger as get_logger

from general_python.algebra.utils import _JAX_AVAILABLE, _KEY, DEFAULT_BACKEND, get_backend as __backend, maybe_jit

# ##################################################################################################
# Test the algebra module
####################################################################################################

from . import linalg as LinalgModule

class AlgebraTests:
    '''
    This is a class that implements the test for algebra module in Python.
    '''
    
    def __init__(self, backend="default"):
        ''' Load the algebra module and set the backend. '''
        
        if not isinstance(backend, str):
            raise ValueError("Backend must be a string.")
        if backend.lower() not in ["default", "np", "jnp", "jax", "numpy"]:
            raise ValueError(f"Invalid backend specifier: {backend} - must be 'default', 'np', or 'jnp'.")
        
        self.backend    = backend
        self.test_count = 0
        self.logger     = get_logger()

    def _log(self, message, test_number, color="white"):
        # Add test_number to log record
        self.logger.say(self.logger.colorize(f"[TEST {test_number}] {message}", color), log = 0, lvl = 1)

    #! ============================================================================
    # General algebra tests - change of basis, outer product, ket-bra product
    #! ============================================================================

    # -----------------------------------------------------------------------------
    # 1) Change of basis tests
    # -----------------------------------------------------------------------------

    def change_basis(self,
                    U    : 'array-like' = None, 
                    vec  : 'array-like' = None, 
                    tvec : 'array-like' = None, test_number = 1, verbose = False):
        """
        Perform a change of basis on a vector using a transformation matrix and verify the result with tests.
        If U, vec, and tvec are not provided, the function uses default values (a swap matrix and corresponding vectors) to
        validate the change of basis operation.
        Parameters:
            U (Union[np.ndarray, None]): The transformation matrix to apply. If None, a default swap matrix [[0, 1], [1, 0]] is used.
            vec (Union[np.ndarray, None]): The original vector to be transformed. If None, a default vector [1, 0] is used.
            tvec (Union[np.ndarray, None]): The expected transformed vector. If None, a default vector [0, 1] is used.
            test_number (int, optional): An identifier for the current test instance, used for logging purposes. Defaults to 1.
            verbose (bool, optional): If True, prints detailed outputs of matrices and vectors involved in the transformation. Defaults to False.
        Raises:
            AssertionError: If the transformed vector does not match the expected vector in the swap test, or if applying the 
                            inverse transformation does not retrieve the original vector.
        Returns:
            None
        """
        backend = __backend(self.backend)
        
        self._log("Starting change_basis", test_number, color = "blue")

        # Test change_basis function with swap matrix and vector
        if U is None and vec is None and tvec is None:
            U   = backend.array([[0, 1], [1, 0]])
            vec = backend.array([1, 0])
            tvec= backend.array([0, 1])
                
        transformed_vec = LinalgModule.change_basis(unitary_matrix=U, state_vector=vec, backend=self.backend)
        assert backend.allclose(transformed_vec, tvec), "change_basis swap test failed"

        # Additional test: check that double transformation retrieves the original vector.
        transformed_twice = LinalgModule.change_basis(unitary_matrix=U.T, state_vector=transformed_vec, backend=self.backend)
        assert backend.allclose(transformed_twice, vec), "change_basis twice test failed"

        if verbose:
            print("U=")
            MatrixPrinter.print_matrix(U)
            print("vec=")
            MatrixPrinter.print_vector(vec)
            print("tvec=")
            MatrixPrinter.print_vector(tvec)
            print("transformed_vec=")
            MatrixPrinter.print_vector(transformed_vec)
            print("transformed_twice=")
            MatrixPrinter.print_vector(transformed_twice)
            
        self._log("Completed change_basis", test_number, color="green")

    # -----------------------------------------------------------------------------
    # 2) Change of basis for matrices tests
    # -----------------------------------------------------------------------------
    
    def change_basis_matrix(self,
                            U : 'array-like' = None,
                            A : 'array-like' = None,
                            At: 'array-like' = None, test_number = 2, verbose = False):
        '''
        Tests the change of basis for matrices.
        '''
        self._log("Starting change_basis_matrix", test_number, color="blue")
        backend = __backend(self.backend)

        # Test change_basis_matrix function: reverse a symmetric matrix.
        if U is None and A is None and At is None:
            U = backend.array([[0, 1], [1, 0]])
            A = backend.array([[1, 2], [3, 4]])
            At= backend.array([[4, 3], [2, 1]])
        
        transformed_matrix  = LinalgModule.change_basis_matrix(U, A)
        assert backend.allclose(transformed_matrix, At), "change_basis_matrix basic test failed"

        # Additional test: check with identity matrix remains unchanged
        transformed_matrix_identity = LinalgModule.change_basis_matrix(U, transformed_matrix, back = True)
        assert backend.allclose(transformed_matrix_identity, A), "change_basis_matrix identity test failed"

        if verbose:
            print("U=")
            MatrixPrinter.print_matrix(U)
            print("A=")
            MatrixPrinter.print_matrix(A)
            print("At=")
            MatrixPrinter.print_matrix(At)
            print("transformed_matrix=")
            MatrixPrinter.print_matrix(transformed_matrix)
        self._log("Completed change_basis_matrix", test_number, color="green")

    # -----------------------------------------------------------------------------
    # 3) Outer product tests
    # -----------------------------------------------------------------------------
    
    def outer(self, A = None, B = None, expAB = None, A_mat = None, B_mat = None, test_number = 3, verbose = False):
        '''
        Tests the outer product of vectors.
        '''
        self._log("Starting outer", test_number, color="blue")
        backend = __backend(self.backend)

        # Test outer function for simple 2D vectors
        if A is None and B is None and expAB is None:
            A       = backend.array([1, 2])
            B       = backend.array([3, 4])
            expAB   = backend.array([[3, 4], [6, 8]])
        if A_mat is None and B_mat is None:
            A_mat = backend.array([[1, 2], [2, 0]])
            B_mat = backend.array([[3, 4], [4, 0]])
        
        outer_product   = LinalgModule.outer(A, B)
        assert backend.allclose(outer_product, expAB), "outer basic test failed"

        # Additional test: outer product for zero vector should be a zero matrix
        A_zero          = backend.array([0, 0])
        outer_zero      = LinalgModule.outer(A_zero, B)
        expected_zero   = backend.zeros((A_zero.shape[0], A_zero.shape[0]))
        assert backend.allclose(outer_zero, expected_zero), "outer zero vector test failed"

        # matrix outer product
        outer_product_mat = LinalgModule.outer(A_mat, B_mat)
        expected_mat      = backend.array([[3, 4, 4, 0], [6, 8, 8, 0], [6, 8, 8, 0], [0, 0, 0, 0]])
        assert backend.allclose(outer_product_mat, expected_mat), "outer matrix test failed"
        
        if verbose:
            print("A=")
            MatrixPrinter.print_vector(A)
            print("B=")
            MatrixPrinter.print_vector(B)
            print("outer_product= A ⊗ B")
            MatrixPrinter.print_matrix(outer_product)
            
            print("A_mat=")
            MatrixPrinter.print_matrix(A_mat)
            print("B_mat=")
            MatrixPrinter.print_matrix(B_mat)
            print("outer_product_mat= A_mat ⊗ B_mat")
            MatrixPrinter.print_matrix(outer_product_mat)

        self._log("Completed outer", test_number, color="green")

    #- -----------------------------------------------------------------------------
    # 4) Kron product tests
    # -----------------------------------------------------------------------------
    
    def kron(self, A = None, B = None, expAB = None, test_number = 4, verbose = False):
        '''
        Tests the Kronecker product of matrices.
        '''
        self._log("Starting kron", test_number, color="blue")
        backend = __backend(self.backend)

        # Test kron function for simple 2D matrices
        if A is None and B is None and expAB is None:
            A       = backend.array([[1, 2], [3, 4]])
            B       = backend.array([[5, 6], [7, 8]])
            expAB   = backend.array([[5, 6, 10, 12], [7, 8, 14, 16], [15, 18, 20, 24], [21, 24, 28, 32]])
        
        kron_product = LinalgModule.kron(A, B)
        assert backend.allclose(kron_product, expAB), "kron basic test failed"

        if verbose:
            print("A=")
            MatrixPrinter.print_matrix(A)
            print("B=")
            MatrixPrinter.print_matrix(B)
            print("kron_product= A ⊗ B")
            MatrixPrinter.print_matrix(kron_product)

        self._log("Completed kron", test_number, color="green")
    
    #- -----------------------------------------------------------------------------
    
    def ket_bra(self, vec = None, test_number = 4, verbose = False):
        '''
        Tests the ket-bra product.
        '''
        self._log("Starting ket_bra", test_number, color="blue")

        # Test ket_bra function with a sample vector
        if vec is None:
            vec = self.backend.array([1, 2])
        
        ket_bra_product     = LinalgModule.ket_bra(vec)
        expected_ket_bra    = self.backend.array([[1, 2], [2, 4]])
        assert self.backend.allclose(ket_bra_product, expected_ket_bra), "ket_bra basic test failed"

        # Additional test: ket_bra of a zero vector should yield a zero matrix
        vec_zero            = self.backend.array([0, 0])
        ket_bra_zero        = LinalgModule.ket_bra(vec_zero)
        expected_zero       = self.backend.zeros((2, 2))
        assert self.backend.allclose(ket_bra_zero, expected_zero), "ket_bra zero vector test failed"

        if verbose:
            MatrixPrinter.print_vector(vec)
            MatrixPrinter.print_matrix(ket_bra_product)

        self._log("Completed ket_bra", test_number, color="green")

    def run_all(self, verbose=False):
        """
        Run all algebra tests and log the execution time.
        This method executes a series of algebra tests by calling various test functions:
            - change_basis
            - change_basis_matrix
        Some other tests (e.g., outer, ket_bra) are currently disabled (commented out).
        It logs the start and completion of the tests with detailed timing information.
        Parameters:
            verbose (bool): If True, enables verbose output during test execution.
        Returns:
            None
        """
        self.logger.say("Starting all algebra tests...", log = 0, lvl = 0)
        
        overall_start   = time.time()

        self.change_basis(verbose=verbose)
        self.change_basis_matrix(verbose=verbose)
        # self.outer(verbose=verbose)
        # self.ket_bra(verbose=verbose)

        overall_end     = time.time()
        self.logger.say(f"All algebra tests completed in {overall_end - overall_start:.4f} seconds!", log=0, lvl=0)

####################################################################################################

def run_algebra_tests(backend='default', verbose=False):
    """
    Run automated tests for algebra functions and classes.
    """
    tests = AlgebraTests(backend)
    tests.run_all(verbose=verbose)
    
####################################################################################################

# TESTING THE SOLVERS OF LINEAR EQUATIONS A X = B

####################################################################################################

class SolversTests:
    """
    This class implements tests for our algebra module.
    It exercises change-of-basis operations and can be extended to test solvers.
    """
    def __init__(self, backend = "default", logger_backend = None):
        valid_backends = ["default", "np", "jnp", "jax", "numpy"]
        if not isinstance(backend, str) or backend.lower() not in valid_backends:
            raise ValueError("Backend must be one of: " + ", ".join(valid_backends))
        self.loggerbackend  = logger_backend
        self.backend        = backend
        self.test_count     = 0
        self.logger         = self.loggerbackend(logfile="algebra_tests.log") if self.loggerbackend is not None else None
        if self.logger is not None:
            self.logger.configure(directory="./logs")
            self.logger.say("Starting algebra tests...", log=0, lvl=1)
            
    def _log(self, message, test_number, color = "white"):
        # Log the test message; you can add formatting here.
        if self.logger is not None:
            self.logger.say(f"[TEST {test_number}] {message}", log=0, lvl=1)
        else:
            print(f"[TEST {test_number}] {message}")

    #! =============================================================================================

    def solver_test(self, make_random = True,
                    symmetric   = True,
                    solver_type = "cg",
                    eps         = 1e-10,
                    max_iter    = 1000,
                    reg         = None,
                    precond_type= None,
                    dtype       = None
                    ):
            """
            Example test for a solver.
            
            Parameters:
                make_random: Whether to generate a random matrix (default: True).
                symmetric: Whether to make the matrix symmetric (default: True).
                solver_type: The type of solver to use (default: "cg").
                eps: The convergence tolerance (default: 1e-10).
                max_iter: The maximum iterations allowed (default: 1000).
                reg: Regularization parameter (default: None).
                precond_type: Type of preconditioner to use (default: None).
            """
            self._log(f"Starting solver test using {solver_type}", self.test_count)
            
            # Generate test matrix A and vector b.
            a, b    = generate_test_mat_vec(make_random, symmetric, dtype=dtype, backend=self.backend)
            
            # Instantiate the chosen solver.
            solver  = choose_solver(solver_type, backend=self.backend, size=a.shape[1], eps=eps, maxiter=max_iter, reg=reg)

            solver.init_from_matrix(a, b, None, reg)
            
            # Create a preconditioner if requested.
            precond = None
            if precond_type is not None:
                precond = choose_precond(precond_type, backend=self.backend)
                precond.set(a=a, sigma=reg, backend=self.backend)
            
            try:
                x, converged = solver.solve(b, x0=None, precond=precond)
            except Exception as e:
                self._log(f"Solver exception: {e}", self.test_count, color="red")
                return

    #! =============================================================================================
    
# --------------------------------------------------------------------------------------------------

__all__ = ['AlgebraTests', 'run_algebra_tests', 'SolversTests']