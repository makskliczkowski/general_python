
import pytest
import numpy as np
from general_python.physics.operators import Operators

class TestOperators:

    def test_resolve_site_basic(self):
        """Test basic site resolution (integer strings)."""
        dim = 10
        assert Operators.resolveSite("0", dim) == 0
        assert Operators.resolveSite("5", dim) == 5
        assert Operators.resolveSite("9", dim) == 9

    def test_resolve_site_special_keywords(self):
        """Test 'L', 'l', 'pi' keywords."""
        dim = 10
        # If OPERATOR_SITE_M_1 is True (default)
        expected_L = dim - 1
        assert Operators.resolveSite("L", dim) == expected_L
        assert Operators.resolveSite("l", dim) == expected_L

        assert Operators.resolveSite("pi", dim) == np.pi

    def test_resolve_site_division(self):
        """Test division syntax 'L_2' -> L/2."""
        dim = 10
        # L_2 means dim / 2
        assert Operators.resolveSite("L_2", dim) == 5
        assert Operators.resolveSite("l_2", dim) == 5

        dim = 11
        # Integer division?
        assert Operators.resolveSite("L_2", dim) == 5

    def test_resolve_site_pi_division(self):
        """Test 'pi_2' -> pi/2."""
        dim = 10
        assert np.isclose(Operators.resolveSite("pi_2", dim), np.pi / 2)
        assert np.isclose(Operators.resolveSite("pi_4", dim), np.pi / 4)

    def test_resolve_site_difference(self):
        """Test difference 'Lm1' -> L-1-1 (if M_1 is True)."""
        # "m" stands for minus?
        # Logic: _dimension - _diff - (1 if M_1 else 0)
        # If site is "Lm1", split by "m" -> "L", "1".
        # Actually split uses OPERATOR_SEP_DIFF="m".
        # If site string contains "m".
        dim = 10
        # "m1" -> dim - 1 - 1 = 8
        assert Operators.resolveSite("m1", dim) == 8
        assert Operators.resolveSite("m0", dim) == 9
        assert Operators.resolveSite("m2", dim) == 7

    def test_resolve_operator(self):
        """Test operator string resolution."""
        dim = 10
        # "Sz/L_2" -> "Sz/5"
        op_str = "Sz/L_2"
        resolved = Operators.resolve_operator(op_str, dim)
        assert resolved == "Sz/5"

        # "N/0" -> "N/0"
        op_str = "N/0"
        resolved = Operators.resolve_operator(op_str, dim)
        assert resolved == "N/0"

        # "Op/m1" -> "Op/8"
        op_str = "Op/m1"
        resolved = Operators.resolve_operator(op_str, dim)
        assert resolved == "Op/8"

    def test_out_of_bounds(self):
        """Test out of bounds site raises Exception."""
        dim = 10
        with pytest.raises(Exception, match="out of range"):
            Operators.resolveSite("10", dim)
        with pytest.raises(Exception, match="out of range"):
            Operators.resolveSite("-1", dim)
