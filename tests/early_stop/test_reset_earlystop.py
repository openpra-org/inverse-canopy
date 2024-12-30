import unittest
import numpy as np
import tensorflow as tf
from inverse_canopy import EarlyStop, ModelParams

class TestEarlyStopResetBehavior(unittest.TestCase):
    def setUp(self):
        self.early_stop = EarlyStop()
        self.initial_params = ModelParams(mus=tf.Variable([1.0, 2.0]), sigmas=tf.Variable([0.1, 0.2]))

    def test_reset_after_no_improvement(self):
        """Test that internal state resets correctly after loss starts improving again."""
        # Initial improvement
        self.early_stop(0.10, step=1, params=self.initial_params)
        self.assertEqual(self.early_stop.best_loss, 0.10)
        self.assertEqual(self.early_stop.step_at_best_loss, 1)
        self.assertEqual(self.early_stop.epochs_since_improvement, 0)

        # No improvement for a few steps
        self.early_stop(0.10, step=2, params=self.initial_params)
        self.early_stop(0.11, step=3, params=self.initial_params)
        self.early_stop(0.12, step=4, params=self.initial_params)
        self.assertEqual(self.early_stop.epochs_since_improvement, 3)

        # Improvement occurs again
        improved_params = ModelParams(mus=tf.Variable([3.0, 4.0]), sigmas=tf.Variable([0.3, 0.4]))
        self.early_stop(0.09, step=5, params=improved_params)
        self.assertEqual(self.early_stop.best_loss, 0.09)
        self.assertEqual(self.early_stop.step_at_best_loss, 5)
        self.assertEqual(self.early_stop.epochs_since_improvement, 0)
        np.testing.assert_allclose(self.early_stop.best_params['mus'].numpy(), [3.0, 4.0], rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(self.early_stop.best_params['sigmas'].numpy(), [0.3, 0.4], rtol=1e-7, atol=1e-9)

if __name__ == '__main__':
    unittest.main()
