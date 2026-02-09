import pytest
import numpy as np
import math
from general_python.maths.math_utils import mod_round, mod_floor, next_power, prev_power, find_maximum_idx

class TestCommonEdgeCases:

    def test_mod_round_negative_divisor(self):
        """Test mod_round with negative divisor."""
        # 5 / -2 = -2.5. 5 % -2 = -1.
        # Implementation: if m < 0: m = m - 1 (since -1 < 0) -> -3.5. int -> -3.
        assert mod_round(5, -2) == -3

        # 4 / -2 = -2.0. 4 % -2 = 0.
        # m < 0. 0 < 0 False. m = m + 1 -> -1.0. int -> -1.
        # Wait, if a % b < 0 is False (0 is not < 0), it goes to else: m + 1.
        assert mod_round(4, -2) == -1

    def test_next_power_edges(self):
        """Test next_power edge cases."""
        # next_power(3, 2) -> 4
        assert next_power(3, 2) == 4

        # next_power(4, 2) -> 4 (log(4)/log(2) = 2. ceil(2)=2. 2^2=4)
        assert next_power(4, 2) == 4

        # next_power(0.5, 2) -> 2^-1 = 0.5?
        # log(0.5)/log(2) = -1. ceil(-1) = -1. 2^-1 = 0.5.
        assert next_power(0.5, 2) == 0.5

        # next_power(0) -> Crash?
        with pytest.raises(ValueError):
             # math.log(0) raises ValueError
             next_power(0)

    def test_find_maximum_idx_numpy(self):
        """Test find_maximum_idx with numpy array."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        # Axis 1
        idx = find_maximum_idx(arr)
        np.testing.assert_array_equal(idx, [2, 2])

        # 1D array?
        # Implementation: np.argmax(x, axis=1).
        # If x is 1D, axis 1 fails.
        arr1d = np.array([1, 2, 3])
        try:
             from numpy.exceptions import AxisError
        except ImportError:
             AxisError = np.AxisError

        with pytest.raises(AxisError):
             find_maximum_idx(arr1d)

    def test_mod_funcs_zero_division(self):
        """Test zero division behavior."""
        for func in [mod_round, mod_floor]:
            with pytest.raises(ValueError, match="Divisor 'b' cannot be zero"):
                func(5, 0)
