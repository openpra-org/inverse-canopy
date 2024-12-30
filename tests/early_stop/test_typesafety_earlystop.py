import unittest
import numpy as np
import tensorflow as tf
from inverse_canopy import EarlyStop, ModelParams

class TestEarlyStopTypeAndValueSafety(unittest.TestCase):
    def setUp(self):
        self.early_stop = EarlyStop()
        self.valid_params = ModelParams(mus=tf.Variable([1.0, 2.0]), sigmas=tf.Variable([0.1, 0.2]))

    def test_invalid_loss_type(self):
        """Test passing an invalid type for current_loss."""
        with self.assertRaises(TypeError):
            self.early_stop("not_a_float", step=1, params=self.valid_params)

    def test_invalid_step_type(self):
        """Test passing an invalid type for step."""
        with self.assertRaises(TypeError):
            self.early_stop(0.05, step="not_an_int", params=self.valid_params)

    def test_invalid_params_type(self):
        """Test passing an invalid type for params."""
        with self.assertRaises(TypeError):
            self.early_stop(0.05, step=1, params="not_a_ModelParams")

    def test_negative_loss(self):
        """Test passing a negative value for current_loss."""
        self.early_stop(-0.01, step=1, params=self.valid_params)
        self.assertEqual(self.early_stop.best_loss, -0.01)
        self.assertEqual(self.early_stop.step_at_best_loss, 1)

    def test_zero_min_delta(self):
        """Test behavior with min_delta set to zero."""
        early_stop_zero_delta = EarlyStop(min_delta=0)
        early_stop_zero_delta(0.05, step=1, params=self.valid_params)
        early_stop_zero_delta(0.05, step=2, params=self.valid_params)
        self.assertEqual(early_stop_zero_delta.epochs_since_improvement, 1)

    def test_zero_patience(self):
        """Test behavior with patience set to zero."""
        early_stop_zero_patience = EarlyStop(patience=0)
        early_stop_zero_patience(0.05, step=1, params=self.valid_params)
        early_stop_zero_patience(0.06, step=2, params=self.valid_params)
        self.assertTrue(early_stop_zero_patience.should_stop)

    def test_large_values(self):
        """Test with very large values of current_loss."""
        large_loss = 1e10
        self.early_stop(large_loss, step=1, params=self.valid_params)
        self.assertEqual(self.early_stop.best_loss, large_loss)

    def test_small_values(self):
        """Test with very small values of current_loss."""
        small_loss = 1e-10
        self.early_stop(small_loss, step=1, params=self.valid_params)
        self.assertEqual(self.early_stop.best_loss, small_loss)

    def test_nan_loss(self):
        """Test behavior when current_loss is NaN."""
        self.early_stop(np.nan, step=1, params=self.valid_params)
        self.assertTrue(np.isnan(self.early_stop.best_loss))

    def test_inf_loss(self):
        """Test behavior when current_loss is infinity."""
        self.early_stop(np.inf, step=1, params=self.valid_params)
        self.assertEqual(self.early_stop.best_loss, np.inf)

if __name__ == '__main__':
    unittest.main()