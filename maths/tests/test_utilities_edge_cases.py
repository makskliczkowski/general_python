"""Regression coverage for small ``math_utils`` helper functions."""

import numpy as np
import pytest

from general_python.maths.math_utils import (
    Fitter,
    find_nearest_idx,
    find_nearest_val,
    mod_ceil,
    mod_euc,
    mod_floor,
    mod_round,
    next_power,
    prev_power,
)


class TestMathUtilities:

    def test_find_nearest(self):
        """Test finding nearest value/index in array."""
        arr = np.array([1.0, 2.0, 5.0, 10.0])

        val = find_nearest_val(arr, 4.9, None)
        assert val == 5.0
        assert find_nearest_idx(arr, 4.9) == 2

        val = find_nearest_val(arr, 1.6, None)
        assert val == 2.0
        assert find_nearest_idx(arr, 1.6) == 1

        val = find_nearest_val(arr, 10.0, None)
        assert val == 10.0
        assert find_nearest_idx(arr, 10.0) == 3

    def test_powers(self):
        """Test next/prev power functions."""
        assert next_power(3) == 4
        assert next_power(4) == 4
        assert next_power(5) == 8

        assert prev_power(3) == 2
        assert prev_power(4) == 4
        assert prev_power(5) == 4

        assert next_power(90, base=10) == 100
        assert prev_power(90, base=10) == 10

    def test_mod_functions(self):
        """Test custom modulo functions."""
        assert mod_euc(-1, 3) == 2
        assert mod_euc(1, 3) == 1
        assert mod_euc(-4, 3) == 2

        assert mod_floor(5, 2) == 2
        assert mod_floor(-5, 2) == -4

        assert mod_ceil(5, 2) == 3

        assert mod_round(5, 2) == 2

    def test_fitter_basic(self):
        """Test basic Fitter usage."""
        x = np.array([0, 1, 2, 3])
        y = 2 * x + 1

        fit_params = Fitter.fitLinear(x, y)
        popt = fit_params.popt

        assert np.isclose(popt[0], 2.0)
        assert np.isclose(popt[1], 1.0)

        assert np.isclose(fit_params(4), 9.0)

    def test_fitter_exp(self):
        """Test exponential fit."""
        x = np.linspace(0, 2, 10)
        y = 2.0 * np.exp(-3.0 * x)

        fit_params = Fitter.fitExp(x, y)
        popt = fit_params.popt
        assert np.isclose(popt[0], 2.0, rtol=1e-3)
        assert np.isclose(popt[1], 3.0, rtol=1e-3)
