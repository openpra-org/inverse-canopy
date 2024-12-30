import unittest
import numpy as np
import tensorflow as tf
from inverse_canopy.lognormal_utils import compute_mu_sigma


class TestComputeMuSigma(unittest.TestCase):
    def test_basic_functionality(self):
        """Test basic functionality with known values."""
        mean = tf.constant(10.0, dtype=tf.float64)
        std = tf.constant(2.0, dtype=tf.float64)
        mu, sigma = compute_mu_sigma(mean, std)
        # Expected values calculated manually or from a reliable source
        expected_mu = np.log(10**2 / np.sqrt(2**2 + 10**2))
        expected_sigma = np.sqrt(np.log(1 + (2**2 / 10**2)))
        self.assertAlmostEqual(mu.numpy(), expected_mu, places=5)
        self.assertAlmostEqual(sigma.numpy(), expected_sigma, places=5)

    def test_negative_or_zero_values(self):
        """Test behavior with negative or zero mean and standard deviation."""
        mean = tf.constant(0.0, dtype=tf.float64)
        std = tf.constant(2.0, dtype=tf.float64)
        mu, sigma = compute_mu_sigma(mean, std)
        self.assertEqual(mu, float("-inf"))
        self.assertEqual(sigma, float("inf"))

        mean = tf.constant(10.0, dtype=tf.float64)
        std = tf.constant(0.0, dtype=tf.float64)
        mu, sigma = compute_mu_sigma(mean, std)
        expected_mu = np.log(10)
        expected_sigma = 0
        self.assertAlmostEqual(mu.numpy(), expected_mu, places=5)
        self.assertAlmostEqual(sigma.numpy(), expected_sigma, places=5)

    def test_type_and_shape(self):
        """Test handling of different data types and shapes."""
        mean = tf.constant([10.0, 20.0], dtype=tf.float32)
        std = tf.constant([2.0, 4.0], dtype=tf.float32)
        mu, sigma = compute_mu_sigma(mean, std, dtype=tf.float32)
        self.assertIsInstance(mu, tf.Tensor)
        self.assertIsInstance(sigma, tf.Tensor)
        self.assertEqual(mu.dtype, tf.float32)
        self.assertEqual(sigma.dtype, tf.float32)
        self.assertEqual(mu.shape, mean.shape)
        self.assertEqual(sigma.shape, std.shape)

    def test_edge_cases(self):
        """Test with tiny and large values."""
        mean = tf.constant([1e-10, 1e10], dtype=tf.float64)
        std = tf.constant([1e-10, 1e10], dtype=tf.float64)
        mu, sigma = compute_mu_sigma(mean, std)
        self.assertTrue(np.all(np.isfinite(mu.numpy())))
        self.assertTrue(np.all(np.isfinite(sigma.numpy())))

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