import unittest
import tensorflow as tf


class TestLogNormalParameters(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64
        self.epsilon = tf.constant(1e-20, dtype=self.dtype)  # Tolerance for floating point comparisons

    def assertTensorsClose(self, a, b, msg=None):
        self.assertTrue(tf.reduce_all(tf.abs(a - b) < self.epsilon), msg=msg)

    # Add more tests for the other methods following a similar pattern


if __name__ == '__main__':
    unittest.main()
