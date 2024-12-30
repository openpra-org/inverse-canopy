import unittest
import tensorflow as tf
from inverse_canopy.lognormal_utils import compute_5th_mean_95th_from_mu_sigma

class TestCompute5thMean95thFromMuSigma(unittest.TestCase):

    def test_basic_functionality(self):
        """Test the function with basic valid inputs."""
        mu = tf.constant(0.0, dtype=tf.float64)
        sigma = tf.constant(0.25, dtype=tf.float64)
        p05, mean, p95 = compute_5th_mean_95th_from_mu_sigma(mu, sigma)
        z = tf.cast(1.6448536269514722, dtype=tf.float64)
        two = tf.cast(2.0, dtype=tf.float64)
        # Expected values calculated manually or from a trusted source
        expected_p05 = tf.exp(mu - z * sigma)
        expected_mean = tf.exp(mu + sigma**two / two)
        expected_p95 = tf.exp(mu + z * sigma)

        self.assertAlmostEqual(p05.numpy(), expected_p05.numpy(), places=5)
        self.assertAlmostEqual(mean.numpy(), expected_mean.numpy(), places=5)
        self.assertAlmostEqual(p95.numpy(), expected_p95.numpy(), places=5)

    def test_negative_sigma_values(self):
        """Test the function with negative sigma values."""
        mu = tf.constant(0.0, dtype=tf.float64)
        sigma = tf.constant(-0.25, dtype=tf.float64)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            compute_5th_mean_95th_from_mu_sigma(mu, sigma)

    def test_type_and_shape(self):
        """Test the function with different input types and shapes."""
        mu = tf.constant([0.0, 0.1], dtype=tf.float32)
        sigma = tf.constant([0.25, 0.35], dtype=tf.float32)
        p05, mean, p95 = compute_5th_mean_95th_from_mu_sigma(mu, sigma, dtype=tf.float32)

        self.assertEqual(p05.dtype, tf.float32)
        self.assertEqual(mean.dtype, tf.float32)
        self.assertEqual(p95.dtype, tf.float32)
        self.assertEqual(p05.shape, mean.shape)
        self.assertEqual(mean.shape, p95.shape)

    def test_edge_cases(self):
        """Test the function with extreme values of mu and sigma."""
        mu = tf.constant([100.0, -100.0], dtype=tf.float64)
        sigma = tf.constant([0.01, 10.0], dtype=tf.float64)
        p05, mean, p95 = compute_5th_mean_95th_from_mu_sigma(mu, sigma)

        # Check if the outputs are finite
        self.assertTrue(tf.reduce_all(tf.math.is_finite(p05)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(mean)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(p95)))

if __name__ == '__main__':
    unittest.main()