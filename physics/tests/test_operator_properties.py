"""
Tests for operator algebra and entropy functions.

Validates operator commutation relations, entropy scaling, and parsing logic.
"""

import pytest
import numpy as np
from general_python.physics.operators import Operators
from general_python.physics.entropy import purity, vn_entropy, Entanglement, entropy

class TestOperatorProperties:

    def test_pauli_commutators(self):
        """Verify Pauli matrices satisfy [S_i, S_j] = i epsilon_ijk S_k."""
        # Manually define Pauli matrices (since they are not exported directly)
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

    def test_entropy_pure_state(self):
        """Test entropy functions on a pure state."""
        # Pure state |0> -> rho = |0><0| -> diag(1, 0)
        rho_pure = np.array([1.0, 0.0])

        # Purity = Tr(rho^2) = 1^2 + 0^2 = 1
        assert np.isclose(purity(rho_pure), 1.0)

        # VN Entropy = -Tr(rho ln rho) = -(1 ln 1 + 0 ln 0) = 0
        assert np.isclose(vn_entropy(rho_pure), 0.0)

        # Wrapper function
        assert np.isclose(entropy(rho_pure, typek=Entanglement.VN), 0.0)

    def test_entropy_mixed_state(self):
        """Test entropy functions on a maximally mixed state."""
        # Mixed state rho = 0.5 * I -> diag(0.5, 0.5)
        rho_mixed = np.array([0.5, 0.5])

        # Purity = 0.5^2 + 0.5^2 = 0.5
        assert np.isclose(purity(rho_mixed), 0.5)

        # VN Entropy = -2 * (0.5 ln 0.5) = -ln(0.5) = ln(2)
        assert np.isclose(vn_entropy(rho_mixed), np.log(2))

        # Renyi Entropy (q=2) = -ln(Tr(rho^2)) = -ln(0.5) = ln(2)
        s_renyi = entropy(rho_mixed, q=2.0, typek=Entanglement.RENYI)
        assert np.isclose(s_renyi, np.log(2))

    def test_operator_parsing_edge_cases(self):
        """Test robust operator parsing."""
        dim = 10

        # Standard parsing
        assert Operators.resolve_operator("Sz/0", dim) == "Sz/0"

        # Arithmetic parsing
        # "L_2" -> L/2 = 5
        assert Operators.resolve_operator("Sp/L_2", dim) == "Sp/5"

        # "m1" -> L-1-1 = 8
        assert Operators.resolve_operator("Sm/m1", dim) == "Sm/8"

        # Invalid input should raise error (handled by resolveSite usually)
        with pytest.raises(Exception):
            Operators.resolve_operator("Sz/20", dim)
