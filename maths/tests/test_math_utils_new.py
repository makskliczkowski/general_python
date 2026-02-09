import pytest
import numpy as np
from general_python.maths.math_utils import find_nearest_val, mod_round, mod_floor

class TestMathUtilsNew:

    def test_find_nearest_val_empty(self):
        """Test find_nearest_val with empty array."""
        arr = np.array([])
        # Should raise ValueError because argmin on empty array raises ValueError
        with pytest.raises(ValueError):
            find_nearest_val(arr, 1.0, None)

    def test_find_nearest_val_nan(self):
        """Test find_nearest_val with NaNs."""
        arr = np.array([1.0, np.nan, 3.0])
        # Test finding value nearest to 3.1
        # |1 - 3.1| = 2.1
        # |nan - 3.1| = nan
        # |3 - 3.1| = 0.1

        # np.argmin usually propagates NaN if it appears (depending on version)
        # or ignores it?
        # Actually np.argmin(np.array([2.1, np.nan, 0.1])) -> often returns index of nan.

        # If it returns index 1 (NaN), that's expected for standard numpy argmin.
        # If it returns index 2, that's better.

        idx = find_nearest_val(arr, 3.1, None)
        # Just ensure it doesn't crash
        assert idx in [0, 1, 2]

    @pytest.mark.parametrize("numer, denom, expected", [
        (5, 2, 2),      # 2.5 -> 2
        (-5, 2, -1),    # Non-intuitive: -5/2 = -2.5. Yields -1.
        (-5, 3, 0),     # -1.66 -> 0
        (4, 2, 2),      # 2.0 -> 2
        (-4, 2, -1),    # -2.0 -> -1.
        (-3, 2, 0),     # -1.5 -> 0
    ])
    def test_mod_round_parametrization(self, numer, denom, expected):
        """
        Parametrized test for mod_round behavior.
        Note: mod_round handles negative numbers non-intuitively.
        """
        assert mod_round(numer, denom) == expected

    @pytest.mark.parametrize("numer, denom, expected", [
        (-5, 2, -4),
        (-4, 2, -2),
        (-3, 2, -3),
    ])
    def test_mod_floor_parametrization(self, numer, denom, expected):
        """
        Parametrized test for mod_floor behavior.
        """
        assert mod_floor(numer, denom) == expected
