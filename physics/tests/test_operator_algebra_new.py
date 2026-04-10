"""Regression checks for operator parsing and entropy helper identities."""

import numpy as np
from general_python.physics.entropy import Entanglement, entropy, purity, vn_entropy
from general_python.physics.operators import Operators


class TestOperatorAlgebra:

    def test_pauli_commutators(self):
        """Verify Pauli matrices satisfy [S_i, S_j] = i epsilon_ijk S_k."""
        # Manually define Pauli matrices (since they are not exported)
        # S = 0.5 * sigma
        Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
        Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
        Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)

        # [Sx, Sy] = Sx Sy - Sy Sx = i Sz
        comm_xy = Sx @ Sy - Sy @ Sx
        expected_xy = 1j * Sz
        np.testing.assert_allclose(comm_xy, expected_xy, atol=1e-10)

        # [Sy, Sz] = i Sx
        comm_yz = Sy @ Sz - Sz @ Sy
        expected_yz = 1j * Sx
        np.testing.assert_allclose(comm_yz, expected_yz, atol=1e-10)

        # [Sz, Sx] = i Sy
        comm_zx = Sz @ Sx - Sx @ Sz
        expected_zx = 1j * Sy
        np.testing.assert_allclose(comm_zx, expected_zx, atol=1e-10)

    def test_operators_parsing(self):
        """Test operator parsing for standard operators."""
        dim = 10

        op = Operators.resolve_operator("Sz/0", dim)
        assert op == "Sz/0"

        op2 = Operators.resolve_operator("Sp/L_2", dim)
        assert op2 == "Sp/5"

    def test_entropy_functions(self):
        """Test entropy calculation functions."""
        rho_pure = np.array([1.0, 0.0])
        assert np.isclose(purity(rho_pure), 1.0)
        assert np.isclose(vn_entropy(rho_pure), 0.0)

        rho_mixed = np.array([0.5, 0.5])
        assert np.isclose(purity(rho_mixed), 0.5)
        assert np.isclose(vn_entropy(rho_mixed), np.log(2))

        s_vn = entropy(rho_mixed, typek=Entanglement.VN)
        assert np.isclose(s_vn, np.log(2))

        s_renyi = entropy(rho_mixed, q=2.0, typek=Entanglement.RENYI)
        assert np.isclose(s_renyi, np.log(2))
