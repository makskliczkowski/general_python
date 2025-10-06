'''
file    : general_python/algebra/solvers/arnoldi.py
author  : Maksymilian Kliczkowski
desc    : Arnoldi and Lanczos iteration implementation.
'''

from typing import Optional, Callable, Tuple
import numpy as np

# Inherit from Solver to reuse setup, though its role is different
from ...algebra.solver import SolverType, Solver, SolverError, SolverErrorMsg
from ...algebra.utils import get_backend
from ...algebra.preconditioners import Preconditioner

# ##############################################################################
# Arnoldi Iteration Class
# ##############################################################################

class ArnoldiIteration(Solver):
    '''
    Performs Arnoldi iteration to build an orthonormal basis V for the
    Krylov subspace K_m(A, v_1) or K_m(M^{-1}A, M^{-1}r_0), and computes
    the corresponding upper Hessenberg matrix H.

    Satisfies A V_m = V_{m+1} H_m (where H_m is m+1 x m, A is op A-shift*I)
    or M^{-1} A V_m = V_{m+1} H_m if preconditioned.
    If the subspace becomes invariant (breakdown), V and H are truncated.

    If the operator is symmetric, this reduces to the Lanczos iteration,
    and H becomes tridiagonal.

    Note: This class primarily generates the basis (V, H). It does not solve Ax=b
    directly. The results V and H are typically used by other methods like GMRES.
    '''

    def __init__(self,
                 backend         ='numpy', # Default to numpy
                 size            = 1,
                 dtype           = np.float64,
                 eps             = 1e-12, # Tolerance for breakdown check
                 maxiter         = 100,   # Dimension of Krylov subspace (m)
                 reg             = None,  # Shift sigma for (A - sigma*I)
                 precond         = None,
                 reorthogonalize = False, # Reorthogonalize basis vectors?
                 symmetric       = False): # Hint if operator is known symmetric (enables Lanczos checks)

        # Force backend to numpy or check compatibility later
        if backend.lower() not in ['numpy', 'np']:
             print(f"Warning: ArnoldiIteration currently implemented for NumPy. Backend set to 'numpy'.")
             backend = 'numpy'
        # Note: restart/maxrestarts from Solver base are not used by Arnoldi itself
        super().__init__(backend, size, dtype, eps, maxiter, reg, precond, restart=False, maxrestarts=1)

        # Use the provided symmetric flag, but Arnoldi works regardless
        self._symmetric   = symmetric
        self._solver_type = SolverType.ARNOLDI
        self._shift = self._reg if self._reg is not None else 0.0

        # Arnoldi specific attributes
        self.reorthogonalize = reorthogonalize # Flag for reorthogonalization
        self.krylov_dim = 0      # Current dimension of the subspace built (k+1)
        self.invariant = False   # Flag if subspace becomes invariant (breakdown)
        self.breakdown = False   # Alias for invariant

        # Basis matrices and vectors (will be initialized in init/iterate)
        self.V: Optional[np.ndarray] = None  # Orthonormal basis matrix V (n x m)
        self.H: Optional[np.ndarray] = None  # Upper Hessenberg matrix H (m+1 x m or m x m)
        self.beta: float = 0.0 # Norm of the initial residual vector r0 or M^{-1}r0

        # Internal work vectors (optional to store, useful for debugging)
        self._w: Optional[np.ndarray] = None # Stores A*v_k or M^-1*A*v_k before orthogonalization

    # -------------------------------------------------------------------------

    def _apply_op(self, v: np.ndarray) -> np.ndarray:
        """Applies the operator (A - shift*I)."""
        if self._mat_vec_mult is None:
            raise SolverError(SolverErrorMsg.MATMULT_NOT_SET)
        # Assumes _mat_vec_mult handles the shift internally
        return self._mat_vec_mult(v, self._shift)

    # -------------------------------------------------------------------------

    def init(self, b: np.ndarray, x0: Optional[np.ndarray] = None) -> None:
        """
        Initializes the Arnoldi iteration.

        Uses 'b' and 'x0' to compute the initial residual r0 = b - A*x0,
        which serves as the source for the starting vector v1.
        If x0 is None, uses b directly.
        """
        # Use Solver's init for basic setup (_n, _dtype, _backend, _solution=x0)
        super().init(b, x0)
        bk = self._backend # NumPy

        if self._n == 0:
            raise ValueError("Arnoldi.init: Input vector 'b' has zero size.")
        if self._maxiter is None or self._maxiter == 0:
            self._maxiter = self._n
        if self._maxiter > self._n:
            print(f"Warning: Arnoldi maxiter ({self._maxiter}) > matrix size ({self._n}). Setting maxiter = {self._n}.")
            self._maxiter = self._n


        # Determine the starting vector source
        if x0 is not None:
            # Initial residual r0 = b - A*x0
            # Use self._solution which holds x0 after super().init
            r0 = b - self._apply_op(self._solution)
        else:
            # Use b directly if no initial guess
            r0 = b.copy()

        # Apply preconditioner M^{-1} to starting vector if needed
        if self._preconditioner:
            if self._shift != 0.0:
                 self.set_preconditioner_sigma(self._shift) # Inform preconditioner
            v1_unnormalized = self._preconditioner(r0)
        else:
            v1_unnormalized = r0

        # Normalize the starting vector v1
        self.beta = bk.linalg.norm(v1_unnormalized) # beta = || M^{-1} r0 || or ||r0||

        # Initialize storage for V and H
        # V: n x maxiter, H: (maxiter+1) x maxiter
        self.V = bk.zeros((self._n, self._maxiter), dtype=self._dtype)
        self.H = bk.zeros((self._maxiter + 1, self._maxiter), dtype=self._dtype)
        self.krylov_dim = 0 # Will be 1 after setting V[:,0]
        self.invariant = False
        self.breakdown = False
        self._iter = 0 # Reset iteration counter used by advance/iterate

        # Handle zero starting vector
        if self.beta < self._eps: # Use provided tolerance
             self.invariant = True
             self.breakdown = True
             print(f"Warning: Arnoldi initial vector norm ({self.beta:.2e}) is close to zero. Subspace is invariant.")
             # V and H remain zero matrices of correct shape, krylov_dim = 0
             return

        # Set first basis vector
        v1 = v1_unnormalized / self.beta
        self.V[:, 0] = v1
        self.krylov_dim = 1

    # -------------------------------------------------------------------------

    def advance(self) -> bool:
        """
        Performs one step of the Arnoldi/Lanczos iteration (k -> k+1).
        Requires init() to have been called.

        Updates self.V and self.H. Increases self.krylov_dim.

        Returns:
            bool: True if the iteration advanced, False if breakdown/invariance
                  occurred or maxiter was reached.
        """
        # Check if we can/should continue
        if self.invariant or self.krylov_dim >= self._maxiter:
            return False

        bk = self._backend
        k = self.krylov_dim - 1 # Current index (0-based) for V, H column k

        # 1. Compute w_unorth = Operator * v_k
        v_k = self.V[:, k]
        w_unorth = self._apply_op(v_k) # w = A' v_k

        # Apply preconditioner M^{-1} if present
        # w = M^{-1} A' v_k
        if self._preconditioner:
             self._w = self._preconditioner(w_unorth)
        else:
             self._w = w_unorth

        # 2. Orthogonalize w against previous basis V_k = [v_0, ..., v_k]
        # Using Modified Gram-Schmidt (MGS)

        w_orth = self._w.copy() # Vector to be orthogonalized
        h_k_col = bk.zeros(self.krylov_dim + 1, dtype=self._dtype) # H[:, k]

        # MGS loop
        for j in range(self.krylov_dim): # Orthogonalize against v_0 to v_k
            v_j = self.V[:, j]
            h_jk = bk.dot(v_j.conj(), w_orth) # Use conj() for complex case
            h_k_col[j] = h_jk
            w_orth -= h_jk * v_j

        # Optional Reorthogonalization (Classical Gram-Schmidt iteration)
        if self.reorthogonalize:
            for j in range(self.krylov_dim):
                 v_j = self.V[:, j]
                 h_corr = bk.dot(v_j.conj(), w_orth)
                 h_k_col[j] += h_corr # Correct H entry
                 w_orth -= h_corr * v_j

        # 3. Compute h_{k+1, k} = ||w_orth||
        h_kp1_k = bk.linalg.norm(w_orth)
        h_k_col[self.krylov_dim] = h_kp1_k # Store in H[k+1, k]

        # Store computed Hessenberg column in H
        # H is (maxiter+1 x maxiter), access H[0 : k+1+1, k]
        self.H[0 : self.krylov_dim + 1, k] = h_k_col

        # Symmetric / Lanczos specific logic from C++ seems unnecessary here.
        # If A' or M^{-1}A' is symmetric, H *should* become tridiagonal naturally
        # through the orthogonalization process (h_jk = 0 for j < k-1).
        # The C++ code seems to enforce this explicitly. MGS handles it implicitly.

        # 4. Check for breakdown/invariance
        if h_kp1_k < self._eps: # Breakdown threshold
            self.invariant = True
            self.breakdown = True
            print(f"Arnoldi breakdown/invariance at iteration {self.krylov_dim}. H({self.krylov_dim},{k}) = {h_kp1_k:.2e}")
            # Optional: Truncate H here? Or do it after iterate() finishes.
            return False # Stop advancing

        # 5. Compute next basis vector v_{k+1}
        v_kp1 = w_orth / h_kp1_k
        # Store in V (n x maxiter), access V[:, k+1]
        self.V[:, self.krylov_dim] = v_kp1
        self.krylov_dim += 1
        self._iter += 1 # Increment internal counter tracking steps done

        return True

    # -------------------------------------------------------------------------

    def iterate(self) -> None:
        """
        Runs the Arnoldi iteration up to self.maxiter steps or until breakdown.
        Requires init() to have been called. Populates self.V and self.H.
        """
        if self.V is None or self.H is None:
             raise RuntimeError("ArnoldiIteration not initialized. Call init() first.")

        # Loop using krylov_dim as the condition
        while self.krylov_dim < self._maxiter:
            can_continue = self.advance()
            if not can_continue:
                break # Stop if breakdown or already at maxiter

        # Final truncation of V and H based on actual Krylov dimension achieved
        final_dim = self.krylov_dim
        self.V = self.V[:, 0:final_dim] # V becomes n x m

        if self.invariant:
            # H should be m x m (square)
            self.H = self.H[0:final_dim, 0:final_dim]
        else:
            # H should be (m+1) x m (rectangular)
            self.H = self.H[0:final_dim + 1, 0:final_dim]

    # -------------------------------------------------------------------------

    # --- Getters for results ---
    def get_V(self) -> Optional[np.ndarray]:
        """Returns the computed orthonormal basis matrix V (n x m)."""
        return self.V

    def get_H(self) -> Optional[np.ndarray]:
        """Returns the computed Hessenberg matrix H (m+1 x m or m x m)."""
        return self.H

    def get_beta(self) -> float:
        """Returns the norm of the initial (preconditioned) residual."""
        return self.beta

    def get_krylov_dim(self) -> int:
        """Returns the final dimension (m) of the Krylov subspace computed."""
        return self.krylov_dim

    # -------------------------------------------------------------------------

    def solve(self, b: np.ndarray, x0: Optional[np.ndarray] = None,
              precond: Optional[Preconditioner] = None) -> np.ndarray:
        '''
        Executes the Arnoldi iteration based on the provided `b` and `x0`.

        This method primarily builds the Arnoldi basis (V, H). It does *not*
        solve the linear system Ax=b itself. The resulting V and H matrices
        are intended to be used by another solver method (e.g., GMRES).

        Parameters:
            b : array-like (n,)
                The right-hand side vector, used to form the initial residual.
            x0 : array-like (n,), optional
                Initial guess for the solution, used for initial residual.
            precond : Preconditioner, optional
                Preconditioner M to use M^{-1}A operator.

        Returns:
            np.ndarray:
                Returns the initial guess `x0` (or zeros if `x0` was None).
                The main results (V, H, beta) are stored as attributes.
        '''
        # Set preconditioner if provided
        if precond is not None:
            self._preconditioner = precond

        # Initialize using b and x0
        self.init(b, x0)

        # Run the iteration
        self.iterate()

        # Return the initial guess - the solution is not computed here
        # self._solution holds x0 from init()
        self._converged = self.invariant # Consider breakdown as "converged" in a sense?
        return self._solution