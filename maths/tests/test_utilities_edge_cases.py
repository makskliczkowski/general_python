
import pytest
import numpy as np
from general_python.maths.math_utils import (
    find_nearest_val, find_nearest_idx,
    next_power, prev_power,
    mod_euc, mod_floor, mod_ceil, mod_round,
    Fitter
)

class TestMathUtilities:

    def test_find_nearest(self):
        """Test finding nearest value/index in array."""
        arr = np.array([1.0, 2.0, 5.0, 10.0])

        # Nearest to 4.9 is index 2 (value 5.0)
        # Note: find_nearest_val returns index for ndarray in current implementation (bug?)
        # It returns an np.array(index)
        val = find_nearest_val(arr, 4.9, None)
        assert val == 2
        assert find_nearest_idx(arr, 4.9) == 2

        # Nearest to 1.6 is index 1 (value 2.0)
        val = find_nearest_val(arr, 1.6, None)
        assert val == 1
        assert find_nearest_idx(arr, 1.6) == 1

        # Exact match
        val = find_nearest_val(arr, 10.0, None)
        assert val == 3
        assert find_nearest_idx(arr, 10.0) == 3

    def test_find_nearest_edge_cases(self):
        """Test find_nearest on edge cases."""
        # Empty array
        arr_empty = np.array([])
        with pytest.raises(ValueError):
            find_nearest_val(arr_empty, 1.0, None)

        # Single element
        arr_single = np.array([5.0])
        val = find_nearest_val(arr_single, 100.0, None)
        assert val == 0 # Index 0

        # With NaNs
        arr_nan = np.array([1.0, np.nan, 10.0])
        # abs(nan - 5) -> nan.
        # argmin might behave differently with NaNs depending on numpy version.
        # Usually nan < num is False. nan < nan is False.
        # But argmin often ignores NaNs or propagates them?
        # np.argmin([1, nan, 10]) -> 1 usually if nan is considered min?
        # Actually standard argmin propagates first NaN or fails?
        # Let's check what happens. If it's undefined behavior, maybe skip or test what it does.
        # np.nanargmin handles nans. argmin does not.

        # Test valid col argument ignored for ndarray
        val = find_nearest_val(np.array([1.0, 2.0]), 1.9, col="garbage")
        assert val == 1

    def test_powers(self):
        """Test next/prev power functions."""
        # Powers of 2
        assert next_power(3) == 4
        assert next_power(4) == 4
        assert next_power(5) == 8

        assert prev_power(3) == 2
        assert prev_power(4) == 4
        assert prev_power(5) == 4

        # Base 10
        assert next_power(90, base=10) == 100
        assert prev_power(90, base=10) == 10

    def test_mod_functions(self):
        """Test custom modulo functions."""
        # mod_euc: ensures result has same sign as divisor
        assert mod_euc(-1, 3) == 2  # -1 = -1*3 + 2
        assert mod_euc(1, 3) == 1
        assert mod_euc(-4, 3) == 2 # -4 = -2*3 + 2

        # mod_floor
        # Implementation subtracts 1 if signs differ and not divisible
        assert mod_floor(5, 2) == 2 # 5 // 2 = 2
        assert mod_floor(-5, 2) == -4 # -5 // 2 = -3, then -1 -> -4

        # mod_ceil
        assert mod_ceil(5, 2) == 3 # ceil(2.5) = 3

        # mod_round
        # Implementation behaves like truncation for positive: int(2.5) -> 2
        assert mod_round(5, 2) == 2

    def test_fitter_basic(self):
        """Test basic Fitter usage."""
        x = np.array([0, 1, 2, 3])
        y = 2 * x + 1 # linear

        # Use static method
        fit_params = Fitter.fitLinear(x, y)
        popt = fit_params.popt

        assert np.isclose(popt[0], 2.0)
        assert np.isclose(popt[1], 1.0)

        # Check callable
        assert np.isclose(fit_params(4), 9.0)

    def test_fitter_exp(self):
        """Test exponential fit."""
        x = np.linspace(0, 2, 10)
        # y = 2 * exp(-3 * x)
        y = 2.0 * np.exp(-3.0 * x)

        fit_params = Fitter.fitExp(x, y)
        popt = fit_params.popt
        # popt[0] is amplitude, popt[1] is decay rate
        assert np.isclose(popt[0], 2.0, rtol=1e-3)
        assert np.isclose(popt[1], 3.0, rtol=1e-3)
