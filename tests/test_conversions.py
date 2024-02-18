import unittest
import tensorflow as tf
from inverse_canopy import compute_mu_sigma, compute_mean_std, compute_mean_error_factor_from_mu_sigma, \
    compute_5th_mean_95th_from_mu_sigma, compute_mu_sigma_from_5th_mean_95th


class TestLogNormalParameters(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64
        self.epsilon = 1e-6  # Tolerance for floating point comparisons

    def assertTensorsClose(self, a, b, msg=None):
        self.assertTrue(tf.reduce_all(tf.abs(a - b) < self.epsilon), msg=msg)

    def test_compute_mu_sigma_base_case(self):
        mean = tf.constant(2.0, dtype=self.dtype)
        std = tf.constant(0.5, dtype=self.dtype)
        mu, sigma = compute_mu_sigma(mean, std)
        expected_mu, expected_sigma = 0.6931471805599453, 0.2876820724517809
        self.assertTensorsClose(mu, tf.constant(expected_mu, dtype=self.dtype))
        self.assertTensorsClose(sigma, tf.constant(expected_sigma, dtype=self.dtype))

    def test_compute_mu_sigma_corner_case(self):
        # Test with mean equal to standard deviation
        mean = tf.constant(1.0, dtype=self.dtype)
        std = tf.constant(1.0, dtype=self.dtype)
        mu, sigma = compute_mu_sigma(mean, std)
        expected_mu, expected_sigma = 0.0, 0.8325546111576977
        self.assertTensorsClose(mu, tf.constant(expected_mu, dtype=self.dtype))
        self.assertTensorsClose(sigma, tf.constant(expected_sigma, dtype=self.dtype))

    # Add more tests for varying dimensions, zero std, large values, etc.

    def test_compute_mean_std_inverse_consistency(self):
        # Test that compute_mean_std is the inverse of compute_mu_sigma
        original_mean = tf.constant([1.5, 2.0, 2.5], dtype=self.dtype)
        original_std = tf.constant([0.1, 0.2, 0.3], dtype=self.dtype)
        mu, sigma = compute_mu_sigma(original_mean, original_std)
        mean, std = compute_mean_std(mu, sigma)
        self.assertTensorsClose(mean, original_mean)
        self.assertTensorsClose(std, original_std)

    # Add more tests for the other methods following a similar pattern


if __name__ == '__main__':
    unittest.main()
