import unittest
import numpy as np
import tensorflow as tf
from inverse_canopy import EarlyStop, ModelParams

class TestEarlyStopBestParamsAndLoss(unittest.TestCase):
    def setUp(self):
        self.early_stop = EarlyStop()
        self.initial_params = ModelParams(mus=tf.Variable([1.0, 2.0]), sigmas=tf.Variable([0.1, 0.2]))

    def test_update_best_loss_and_params_on_improvement(self):
        """Test that best_loss and best_params are updated when an improvement in loss is observed."""
        current_loss = 0.05
        self.early_stop(current_loss, step=1, params=self.initial_params)
        self.assertEqual(self.early_stop.best_loss, current_loss)
        self.assertEqual(self.early_stop.step_at_best_loss, 1)
        np.testing.assert_allclose(self.early_stop.best_params['mus'].numpy(), [1.0, 2.0], rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(self.early_stop.best_params['sigmas'].numpy(), [0.1, 0.2], rtol=1e-7, atol=1e-9)

    def test_no_update_on_worse_loss(self):
        """Test that best_loss and best_params do not update when a worse loss is reported."""
        self.early_stop.best_loss = 0.05
        self.early_stop.best_params = self.initial_params.deep_copy()
        self.early_stop.step_at_best_loss = 1
        worse_loss = 0.1
        self.early_stop(worse_loss, step=2, params=ModelParams(mus=tf.Variable([3.0, 4.0]), sigmas=tf.Variable([0.3, 0.4])))
        self.assertEqual(self.early_stop.best_loss, 0.05)
        self.assertEqual(self.early_stop.step_at_best_loss, 1)
        np.testing.assert_allclose(self.early_stop.best_params['mus'].numpy(), [1.0, 2.0], rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(self.early_stop.best_params['sigmas'].numpy(), [0.1, 0.2], rtol=1e-7, atol=1e-9)

    def test_no_update_on_equal_loss(self):
        """Test that best_loss and best_params do not update when the same loss is reported."""
        self.early_stop.best_loss = 0.05
        self.early_stop.best_params = self.initial_params.deep_copy()
        self.early_stop.step_at_best_loss = 1
        same_loss = 0.05
        self.early_stop(same_loss, step=2, params=ModelParams(mus=tf.Variable([3.0, 4.0]), sigmas=tf.Variable([0.3, 0.4])))
        self.assertEqual(self.early_stop.best_loss, 0.05)
        self.assertEqual(self.early_stop.step_at_best_loss, 1)
        np.testing.assert_allclose(self.early_stop.best_params['mus'].numpy(), [1.0, 2.0], rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(self.early_stop.best_params['sigmas'].numpy(), [0.1, 0.2], rtol=1e-7, atol=1e-9)

if __name__ == '__main__':
    unittest.main()
