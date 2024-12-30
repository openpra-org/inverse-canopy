import unittest
import tensorflow as tf
import numpy as np
from scipy.stats import norm
from inverse_canopy.lognormal_utils import compute_mu_sigma_from_5th_mean_95th

class TestComputeMuSigmaFrom5thMean95th(unittest.TestCase):
    def setUp(self):
        # Set the default dtype for tensors in all tests
        self.dtype = tf.float64

    def test_basic_functionality(self):
        # Known values for 5th percentile, mean, and 95th percentile
        p05 = tf.constant(100.0, dtype=self.dtype)
        mean = tf.constant(200.0, dtype=self.dtype)
        p95 = tf.constant(400.0, dtype=self.dtype)

        # Expected mu and sigma calculated manually or from a reliable source
        expected_mu = tf.constant(5.209526880914445, dtype=self.dtype)
        expected_sigma = tf.constant(0.4214035729169639, dtype=self.dtype)

        # Compute mu and sigma using the function
        mu, sigma = compute_mu_sigma_from_5th_mean_95th(p05, mean, p95, dtype=self.dtype)

        # Check if the computed values are close to the expected values
        self.assertAlmostEqual(mu.numpy(), expected_mu.numpy(), places=5)
        self.assertAlmostEqual(sigma.numpy(), expected_sigma.numpy(), places=5)


    def test_type_and_shape(self):
        # Test with different shapes
        p05 = tf.constant([100.0, 150.0], dtype=self.dtype)
        mean = tf.constant([200.0, 225.0], dtype=self.dtype)
        p95 = tf.constant([400.0, 450.0], dtype=self.dtype)

        # Compute mu and sigma
        mu, sigma = compute_mu_sigma_from_5th_mean_95th(p05, mean, p95, dtype=self.dtype)

        # Check shapes
        self.assertEqual(mu.shape, p05.shape)
        self.assertEqual(sigma.shape, p95.shape)

    def test_edge_cases(self):
        # Test with extreme values
        p05 = tf.constant(1e-10, dtype=self.dtype)
        mean = tf.constant(1e5, dtype=self.dtype)
        p95 = tf.constant(1e10, dtype=self.dtype)

        # Compute mu and sigma
        mu, sigma = compute_mu_sigma_from_5th_mean_95th(p05, mean, p95, dtype=self.dtype)

        # Check for numerical stability (not NaN or inf)
        self.assertFalse(tf.math.is_nan(mu) or tf.math.is_inf(mu))
        self.assertFalse(tf.math.is_nan(sigma) or tf.math.is_inf(sigma))

if __name__ == '__main__':
    unittest.main()