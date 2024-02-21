import unittest
import tensorflow as tf
from inverse_canopy import compute_mu_sigma, compute_mean_std, compute_mean_error_factor_from_mu_sigma, \
    compute_5th_mean_95th_from_mu_sigma, compute_mu_sigma_from_5th_mean_95th


class TestLogNormalParameters(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64
        self.epsilon = tf.constant(1e-20, dtype=self.dtype)  # Tolerance for floating point comparisons

    def assertTensorsClose(self, a, b, msg=None):
        self.assertTrue(tf.reduce_all(tf.abs(a - b) < self.epsilon), msg=msg)

    # Add more tests for the other methods following a similar pattern


if __name__ == '__main__':
    unittest.main()
