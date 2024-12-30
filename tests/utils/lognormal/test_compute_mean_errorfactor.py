import unittest
import tensorflow as tf
from inverse_canopy.lognormal_utils import compute_mean_error_factor_from_mu_sigma

class TestComputeMeanErrorFactorFromMuSigma(unittest.TestCase):
    def test_basic_functionality(self):
        """Test the function with basic valid inputs."""
        mu = tf.constant(0.0, dtype=tf.float64)
        sigma = tf.constant(1.0, dtype=tf.float64)
        two = tf.cast(2.0, dtype=tf.float64)
        z = tf.cast(1.6448536269514722, dtype=tf.float64)
        expected_mean = tf.exp(mu + sigma**two / two)
        expected_error_factor = tf.exp(z * sigma)
        mean, error_factor = compute_mean_error_factor_from_mu_sigma(mu, sigma)
        self.assertAlmostEqual(mean.numpy(), expected_mean.numpy(), places=5)
        self.assertAlmostEqual(error_factor.numpy(), expected_error_factor.numpy(), places=5)

    def test_type_and_shape(self):
        """Test the function with different data types and shapes."""
        mu = tf.constant([0.0, 0.5], dtype=tf.float32)
        sigma = tf.constant([1.0, 1.5], dtype=tf.float32)
        mean, error_factor = compute_mean_error_factor_from_mu_sigma(mu, sigma, dtype=tf.float32)
        self.assertEqual(mean.dtype, tf.float32)
        self.assertEqual(error_factor.dtype, tf.float32)
        self.assertEqual(mean.shape, sigma.shape)
        self.assertEqual(error_factor.shape, sigma.shape)

    def test_edge_cases(self):
        """Test the function with extreme values of mu and sigma."""
        mu = tf.constant([100.0, -100.0], dtype=tf.float64)
        sigma = tf.constant([0.1, 10.0], dtype=tf.float64)
        mean, error_factor = compute_mean_error_factor_from_mu_sigma(mu, sigma)
        # Check if the computation does not result in overflow or underflow
        self.assertFalse(tf.reduce_any(tf.math.is_inf(mean)))
        self.assertFalse(tf.reduce_any(tf.math.is_inf(error_factor)))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(mean)))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(error_factor)))

if __name__ == '__main__':
    unittest.main()