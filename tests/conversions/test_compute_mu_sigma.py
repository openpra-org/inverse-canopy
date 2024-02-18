import unittest
import tensorflow as tf
import numpy as np

from inverse_canopy.lognormal_utils import compute_mu_sigma


class TestComputeMuSigma(unittest.TestCase):
    def test_basic_float32(self):
        """Test with basic float32 inputs."""
        mean = tf.constant(3.0, dtype=tf.float32)
        std = tf.constant(2.0, dtype=tf.float32)
        mu, sigma = compute_mu_sigma(mean, std, dtype=tf.float32)
        # Convert tensors to numpy for assertion
        mu, sigma = mu.numpy(), sigma.numpy()
        self.assertAlmostEqual(mu, 0.9147499, places=5)
        self.assertAlmostEqual(sigma, 0.6064032, places=5)

    def test_basic_float64(self):
        """Test with basic float64 inputs."""
        mean = tf.constant(3.0, dtype=tf.float64)
        std = tf.constant(2.0, dtype=tf.float64)
        mu, sigma = compute_mu_sigma(mean, std, dtype=tf.float64)
        # Convert tensors to numpy for assertion
        mu, sigma = mu.numpy(), sigma.numpy()
        self.assertAlmostEqual(mu, 0.9147498986054511, places=10)
        self.assertAlmostEqual(sigma, 0.6064031498312961, places=10)

    def test_vector_input(self):
        """Test with vector inputs."""
        mean = tf.constant([1.0, 2.0, 3.0], dtype=tf.float64)
        std = tf.constant([0.5, 1.0, 1.5], dtype=tf.float64)
        mu, sigma = compute_mu_sigma(mean, std, dtype=tf.float64)
        # Convert tensors to numpy for assertion
        mu, sigma = mu.numpy(), sigma.numpy()
        expected_mu = np.array([-0.1115717757,  0.5815754049,  0.987040513])
        expected_sigma = np.array([0.4723807271, 0.4723807271, 0.4723807271])
        np.testing.assert_array_almost_equal(mu, expected_mu, decimal=10)
        np.testing.assert_array_almost_equal(sigma, expected_sigma, decimal=10)

    def test_zero_std(self):
        """Test with a standard deviation of zero."""
        mean = tf.constant(3.0, dtype=tf.float64)
        std = tf.constant(0.0, dtype=tf.float64)
        mu, sigma = compute_mu_sigma(mean, std, dtype=tf.float64)
        # Convert tensors to numpy for assertion
        mu, sigma = mu.numpy(), sigma.numpy()
        self.assertAlmostEqual(mu, np.log(3.0), places=10)
        self.assertAlmostEqual(sigma, 0.0, places=10)


if __name__ == '__main__':
    unittest.main()
