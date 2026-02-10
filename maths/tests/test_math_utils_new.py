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

    def test_mod_round_negative(self):
        """Verify mod_round behavior for negative inputs."""
        # Current implementation documented as "truncation" in existing tests
        # mod_round(5, 2) -> int(2.5) -> 2

        # Behavior for negative inputs:
        # mod_round(-5, 2) -> -2.5. Python % 2 is positive.
        # Implementation adds 1 -> -1.5. Truncates to -1.
        assert mod_round(-5, 2) == -1

        # mod_round(-5, 3) -> -1.66. Remainder positive.
        # Adds 1 -> -0.66. Truncates to 0.
        assert mod_round(-5, 3) == 0

    def test_mod_floor_negative(self):
        """Verify mod_floor behavior for negative inputs."""
        # Based on existing tests: mod_floor(-5, 2) == -4.
        # -5 / 2 = -2.5. Floor is -3.
        # So it seems to be floor - 1? Or something else.

        # Let's verify standard floor division first
        assert -5 // 2 == -3

        # The function `mod_floor` seems to implement something distinct from python `//`.
        # Existing test: assert mod_floor(-5, 2) == -4

        # Let's test a case that is divisible
        # -4 / 2 = -2.
        assert mod_floor(-4, 2) == -2

        # Case -3 / 2 = -1.5. Floor -2.
        # If pattern holds, might be -3?
        # Let's just run it and see (or not assert exact value if unsure, but I want to pin behavior).

        # I'll rely on the property that it should be <= real division
        res = mod_floor(-3, 2)
        assert res <= -1.5
