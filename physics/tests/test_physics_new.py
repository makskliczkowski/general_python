
import pytest
import numpy as np

class TestPhysicsNew:

    def test_pauli_algebra(self):
        """
        Verify Pauli matrices satisfy algebraic invariants:
        [sigma_i, sigma_j] = 2i * epsilon_ijk * sigma_k
        {sigma_i, sigma_j} = 2 * delta_ij * I
        sigma_i^dagger = sigma_i
        Tr(sigma_i) = 0
        """
        # Define Pauli matrices
        sigma_0 = np.eye(2)
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

        sigmas = [sigma_x, sigma_y, sigma_z]

        # 1. Hermiticity
        for s in sigmas:
            assert np.allclose(s, s.conj().T)

        # 2. Trace zero
        for s in sigmas:
            assert np.isclose(np.trace(s), 0)

        # 3. Commutation relations
        # [Sx, Sy] = 2i Sz
        assert np.allclose(sigma_x @ sigma_y - sigma_y @ sigma_x, 2j * sigma_z)
        # [Sy, Sz] = 2i Sx
        assert np.allclose(sigma_y @ sigma_z - sigma_z @ sigma_y, 2j * sigma_x)
        # [Sz, Sx] = 2i Sy
        assert np.allclose(sigma_z @ sigma_x - sigma_x @ sigma_z, 2j * sigma_y)

        # 4. Anticommutation relations
        # {Sx, Sy} = 0
        assert np.allclose(sigma_x @ sigma_y + sigma_y @ sigma_x, 0)
        # {Sx, Sx} = 2I
        assert np.allclose(sigma_x @ sigma_x + sigma_x @ sigma_x, 2 * sigma_0)

    def test_density_matrix_properties(self):
        """
        Test properties of a density matrix rho.
        rho = |psi><psi|
        Tr(rho) = 1
        rho^2 = rho (pure state)
        rho^dagger = rho
        """
        # Random state vector
        psi = np.random.randn(2) + 1j * np.random.randn(2)
        psi /= np.linalg.norm(psi)

        rho = np.outer(psi, psi.conj())

        # Trace = 1
        assert np.isclose(np.trace(rho), 1.0)

        # Hermiticity
        assert np.allclose(rho, rho.conj().T)

        # Purity check: Tr(rho^2) = 1 for pure state
        rho2 = rho @ rho
        assert np.isclose(np.trace(rho2), 1.0)
        assert np.allclose(rho, rho2)

    def test_operators_resolve_site_edge_cases(self):
        """
        Additional tests for Operators.resolveSite from physics/operators.py
        covering edge cases not in original test.
        """
        from general_python.physics.operators import Operators

        dim = 10
        # "l" -> dim - 1 = 9
        assert Operators.resolveSite("l", dim) == 9

        # "pi/2"
        # Operators.OPERATOR_PI is "pi"
        # OPERATOR_SEP_DIV is "_"
        # So "pi_2" means pi / 2
        val = Operators.resolveSite("pi_2", dim)
        assert np.isclose(val, np.pi / 2)

        # "L_1" -> L/1 = 10 (if L=10)
        # Note: Operators.OPERATOR_SITEU is "L"
        # resolveSite("L_1", 10) -> "L" / "1".
        # split -> "L", "1".
        # resolveSite("1", 10) -> 1.
        # "L" in site -> dim / 1 = 10.
        assert Operators.resolveSite("L_1", 10) == 10
