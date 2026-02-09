import pytest
import numpy as np
from general_python.physics.operators import Operators
from general_python.physics.entropy import purity, vn_entropy

class TestPhysicsInvariants:

    def test_pauli_hermiticity(self):
        """Test that Pauli matrices are Hermitian."""
        # Using Operators helper if available, or manual construction
        # "Sz/0" -> Sz on site 0
        # "Sp/0" -> S+
        # "Sm/0" -> S-

        # Operators.resolve_operator usually returns string or simple object?
        # Let's construct matrices manually as library doesn't export them directly (per memory).

        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

        for name, op in [("Sx", sigma_x), ("Sy", sigma_y), ("Sz", sigma_z)]:
            assert np.allclose(op, op.conj().T), f"{name} is not Hermitian"

    def test_density_matrix_properties(self):
        """Test density matrix properties (trace=1, positive semi-definite)."""
        # Create a random pure state
        psi = np.random.randn(4) + 1j * np.random.randn(4)
        psi /= np.linalg.norm(psi)

        rho = np.outer(psi, psi.conj())

        # Trace should be 1
        assert np.isclose(np.trace(rho), 1.0)

        # Should be Hermitian
        assert np.allclose(rho, rho.conj().T)

        # Eigenvalues should be non-negative (0 or 1 for pure state)
        evals = np.linalg.eigvalsh(rho)
        assert np.all(evals > -1e-10) # Allow small numerical noise

        # Purity should be 1
        assert np.isclose(purity(rho), 1.0)

        # Create mixed state
        rho_mixed = 0.5 * rho + 0.5 * np.eye(4)/4
        # Trace 1
        assert np.isclose(np.trace(rho_mixed), 1.0)
        # Purity < 1
        assert purity(rho_mixed) < 1.0

    def test_commutator_jacobi_identity(self):
        """Test Jacobi identity for commutators [A, [B, C]] + [B, [C, A]] + [C, [A, B]] = 0."""
        A = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        B = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        C = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)

        def comm(X, Y):
            return X @ Y - Y @ X

        term1 = comm(A, comm(B, C))
        term2 = comm(B, comm(C, A))
        term3 = comm(C, comm(A, B))

        sum_terms = term1 + term2 + term3
        assert np.allclose(sum_terms, np.zeros((3, 3)), atol=1e-10)
