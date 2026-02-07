
import pytest
import numpy as np
from general_python.maths.math_utils import (
    find_nearest_val, find_nearest_idx,
    Fitter, mod_euc, mod_floor, mod_ceil, mod_round, mod_trunc
)

class TestMathsNew:

    def test_find_nearest_val_is_index(self):
        """
        Verify that find_nearest_val returns the INDEX, not the value.
        """
        arr = np.array([10.0, 20.0, 30.0])
        val = 21.0
        # Nearest is 20.0 at index 1
        idx = find_nearest_val(arr, val, None)
        assert idx == 1

        # Verify it works for JAX arrays if available (optional)
        # We just test numpy path here

    def test_find_nearest_edge_cases(self):
        """
        Test find_nearest with edge cases.
        """
        arr = np.array([1.0, 2.0, 3.0])

        # Exact match
        assert find_nearest_val(arr, 1.0, None) == 0

        # Midpoint (implementation dependent, usually first one or depends on rounding)
        # 1.5 is equidistant to 1.0 and 2.0.
        # np.argmin usually picks first occurrence.
        assert find_nearest_val(arr, 1.5, None) == 0 # |1-1.5|=0.5, |2-1.5|=0.5. Index 0 comes first.

    def test_fitter_linear_synthetic(self):
        """
        Test Fitter.fitLinear with synthetic data + noise.
        """
        np.random.seed(42)
        x = np.linspace(0, 10, 20)
        true_a = 2.5
        true_b = -1.0
        y = true_a * x + true_b

        # Perfect fit
        params = Fitter.fitLinear(x, y)
        assert np.allclose(params.popt, [true_a, true_b])

        # With noise
        y_noisy = y + np.random.normal(0, 0.1, size=len(y))
        params_noisy = Fitter.fitLinear(x, y_noisy)
        assert np.allclose(params_noisy.popt, [true_a, true_b], rtol=0.1)

    def test_fitter_histogram_gaussian(self):
        """
        Test fitting a histogram to a Gaussian.
        """
        np.random.seed(42)
        data = np.random.normal(loc=5.0, scale=2.0, size=1000)
        counts, edges = np.histogram(data, bins=20, density=True)
        centers = (edges[:-1] + edges[1:]) / 2

        # Fit
        # fit_histogram returns FitterParams
        # Gaussian params: mu, sigma
        # fit_histogram calls Fitter.gaussian(x, mu, sigma)
        params = Fitter.fit_histogram(edges, counts, typek='gaussian', centers=centers)

        mu_fit, sigma_fit = params.popt
        assert np.isclose(mu_fit, 5.0, rtol=0.2)
        assert np.isclose(sigma_fit, 2.0, rtol=0.2)

    def test_mod_functions_zero_division(self):
        """
        Test that mod functions raise ValueError on zero divisor.
        """
        with pytest.raises(ValueError, match="Divisor 'b' cannot be zero"):
            mod_euc(5, 0)

        with pytest.raises(ValueError, match="Divisor 'b' cannot be zero"):
            mod_floor(5, 0)

        with pytest.raises(ValueError, match="Divisor 'b' cannot be zero"):
            mod_ceil(5, 0)

        with pytest.raises(ValueError, match="Divisor 'b' cannot be zero"):
            mod_round(5, 0)

        with pytest.raises(ValueError, match="Divisor 'b' cannot be zero"):
            mod_trunc(5, 0)
