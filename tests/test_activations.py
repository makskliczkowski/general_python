
import unittest
import numpy as np
import jax.numpy as jnp
from general_python.ml.net_impl.activation_functions import tanh_jnp, tanh, log_cosh_jnp
from general_python.ml.net_impl.networks.net_rbm import RBM

class TestActivations(unittest.TestCase):
    def test_tanh_real_negative(self):
        x = np.array([-1.0, -2.0])
        expected = np.tanh(x)

        # NumPy version
        res_np = tanh(x)
        np.testing.assert_allclose(res_np, expected, rtol=1e-5)

        # JAX version
        res_jax = tanh_jnp(jnp.array(x))
        np.testing.assert_allclose(res_jax, expected, rtol=1e-5)

    def test_tanh_complex(self):
        # tanh(-z) = -tanh(z)
        z = np.array([1.0 + 1.0j])
        z_neg = -z

        expected = np.tanh(z_neg)

        # NumPy version
        res_np = tanh(z_neg)
        np.testing.assert_allclose(res_np, expected, rtol=1e-5)

        # JAX version
        res_jax = tanh_jnp(jnp.array(z_neg))
        np.testing.assert_allclose(res_jax, expected, rtol=1e-5)

    def test_rbm_holomorphic(self):
        # RBM with complex params should be holomorphic
        rbm = RBM(
            input_shape=(4,),
            n_hidden=2,
            param_dtype=jnp.complex64,
            dtype=jnp.complex64,
            seed=42
        )
        rbm.init()
        self.assertTrue(rbm.check_holomorphic())

if __name__ == '__main__':
    unittest.main()
