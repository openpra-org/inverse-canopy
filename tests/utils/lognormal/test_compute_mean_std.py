import unittest
import tensorflow as tf
from tensorflow.python.framework.errors_impl import UnimplementedError

from inverse_canopy.lognormal_utils import compute_mean_std


class TestComputeMeanStd(unittest.TestCase):
    def test_basic_functionality(self):
        """ Test basic functionality with known values """
        mu = tf.constant(0.0, dtype=tf.float64)
        sigma = tf.constant(1.0, dtype=tf.float64)
        two = tf.cast(2.0, dtype=tf.float64)
        one = tf.cast(1.0, dtype=tf.float64)
        expected_mean = tf.exp(mu + sigma ** two / two)
        expected_std = expected_mean * tf.sqrt(tf.exp(sigma ** two) - one)

        mean, std = compute_mean_std(mu, sigma)

        self.assertAlmostEqual(mean.numpy(), expected_mean.numpy(), places=5)
        self.assertAlmostEqual(std.numpy(), expected_std.numpy(), places=5)

    def test_type_and_shape(self):
        """ Test the function with different data types and shapes """
        mu = tf.constant([0.0, 1.0], dtype=tf.float32)
        sigma = tf.constant([1.0, 2.0], dtype=tf.float32)
        mean, std = compute_mean_std(mu, sigma, dtype=tf.float32)

        self.assertEqual(mean.dtype, tf.float32)
        self.assertEqual(std.dtype, tf.float32)
        self.assertEqual(mean.shape, (2,))
        self.assertEqual(std.shape, (2,))

    def test_edge_cases(self):
        """ Test the function with extreme values of mu and sigma """
        mu = tf.constant([100.0, -100.0], dtype=tf.float64)
        sigma = tf.constant([0.1, 0.1], dtype=tf.float64)
        mean, std = compute_mean_std(mu, sigma)

        # Check if the results are finite and not overflowed
        self.assertTrue(tf.reduce_all(tf.math.is_finite(mean)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(std)))

    def test_invalid_input_types(self):
        """ Test the function with invalid input types """
        mu = "0.0"
        sigma = "1.0"
        with self.assertRaises(UnimplementedError):
            mean, std = compute_mean_std(mu, sigma)


if __name__ == '__main__':
    unittest.main()