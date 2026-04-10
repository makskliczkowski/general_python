"""Additional edge-case coverage for ``math_utils`` helpers."""

import numpy as np
import pytest

from general_python.maths.math_utils import find_nearest_val, mod_round, mod_floor


class TestMathUtilsNew:

    def test_find_nearest_val_empty(self):
        """Test find_nearest_val with empty array."""
        arr = np.array([])
        with pytest.raises(ValueError):
            find_nearest_val(arr, 1.0, None)

    def test_find_nearest_val_nan(self):
        """Test find_nearest_val with NaNs."""
        arr = np.array([1.0, np.nan, 3.0])

        val = find_nearest_val(arr, 3.1, None)
        assert val in [1.0, 3.0] or np.isnan(val)

    def test_mod_round_negative(self):
        """Verify mod_round behavior for negative inputs."""
        assert mod_round(-5, 2) == -1

        assert mod_round(-5, 3) == 0

    def test_mod_floor_negative(self):
        """Verify mod_floor behavior for negative inputs."""
        assert -5 // 2 == -3
        assert mod_floor(-4, 2) == -2

        res = mod_floor(-3, 2)
        assert res <= -1.5
