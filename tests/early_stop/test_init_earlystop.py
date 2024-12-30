import unittest
import numpy as np
import tensorflow as tf
from inverse_canopy import EarlyStop, ModelParams

class TestEarlyStopInitialization(unittest.TestCase):
    def setUp(self):
        self.early_stop = EarlyStop()
        self.params = ModelParams(mus=tf.Variable([1.0, 2.0], dtype=tf.float32),
                                          sigmas=tf.Variable([0.1, 0.2], dtype=tf.float32))
    def test_default_initialization(self):
        """Test the default initialization of the EarlyStop class."""
        early_stop = EarlyStop()
        self.assertEqual(early_stop.min_delta, 0.001, "Default min_delta should be 0.001")
        self.assertEqual(early_stop.patience, 10, "Default patience should be 10")
        self.assertIsNone(early_stop.best_loss, "Default best_loss should be None")
        self.assertEqual(early_stop.epochs_since_improvement, 0, "Default epochs_since_improvement should be 0")
        self.assertFalse(early_stop.should_stop, "Default should_stop should be False")
        self.assertTrue(np.isnan(early_stop.step_at_best_loss), "Default step_at_best_loss should be NaN")
        self.assertIsInstance(early_stop.best_params, ModelParams, "best_params should be an instance of ModelParams")

    def test_custom_initialization(self):
        """Test the initialization with custom values."""
        custom_min_delta = 0.01
        custom_patience = 20
        early_stop = EarlyStop(min_delta=custom_min_delta, patience=custom_patience)
        self.assertEqual(early_stop.min_delta, custom_min_delta, "Custom min_delta should be set correctly")
        self.assertEqual(early_stop.patience, custom_patience, "Custom patience should be set correctly")
        self.assertIsNone(early_stop.best_loss, "best_loss should still be None")
        self.assertEqual(early_stop.epochs_since_improvement, 0, "epochs_since_improvement should still be 0")
        self.assertFalse(early_stop.should_stop, "should_stop should still be False")
        self.assertTrue(np.isnan(early_stop.step_at_best_loss), "step_at_best_loss should still be NaN")
        self.assertIsInstance(early_stop.best_params, ModelParams, "best_params should still be an instance of ModelParams")


    def test_min_delta_zero(self):
        """Test behavior when min_delta is set to zero."""
        early_stop = EarlyStop(min_delta=0, patience=10)
        early_stop(0.1, step=1, params=self.params)
        early_stop(0.1, step=2, params=self.params)  # No improvement, but min_delta is zero
        self.assertEqual(early_stop.epochs_since_improvement, 1, "Epochs since improvement should increment even with min_delta=0")

    def test_patience_zero(self):
        """Test behavior when patience is set to zero."""
        early_stop = EarlyStop(min_delta=0.001, patience=0)
        early_stop(0.1, step=1, params=self.params)
        early_stop(0.101, step=2, params=self.params)  # No sufficient improvement
        self.assertTrue(early_stop.should_stop, "Training should stop immediately if patience is zero")

    def test_current_loss_nan(self):
        """Test behavior when current_loss is NaN."""
        early_stop = EarlyStop()
        early_stop(np.nan, step=1, params=self.params)
        self.assertTrue(np.isnan(early_stop.best_loss), "Best loss should be NaN if current_loss is NaN")
        self.assertEqual(early_stop.epochs_since_improvement, 0, "Epochs since improvement should not increment on NaN loss")

    def test_current_loss_inf(self):
        """Test behavior when current_loss is inf."""
        early_stop = EarlyStop()
        early_stop(np.inf, step=1, params=self.params)
        self.assertTrue(np.isinf(early_stop.best_loss), "Best loss should be inf if current_loss is inf")
        self.assertEqual(early_stop.epochs_since_improvement, 0, "Epochs since improvement should not increment on inf loss")

    def test_large_values_for_min_delta_and_patience(self):
        """Test with very large values for min_delta and patience."""
        early_stop = EarlyStop(min_delta=1e10, patience=1e6)
        early_stop(0.1, step=1, params=self.params)
        early_stop(1e10, step=2, params=self.params)  # Huge improvement
        self.assertFalse(early_stop.should_stop, "Should not stop as the improvement is huge")
        self.assertEqual(early_stop.best_loss, 1e10, "Best loss should update to the huge improvement value")

    def test_deep_copy_model_params(self):
        """Test that best_params is a deep copy and does not change with original params."""
        # Initial loss improvement
        initial_loss = 0.05
        self.early_stop(initial_loss, step=1, params=self.params)

        # Modify original parameters after passing to EarlyStop
        self.params['mus'].assign([3.0, 4.0])
        self.params['sigmas'].assign([0.3, 0.4])

        # Check that best_params in EarlyStop did not change
        best_mus = self.early_stop.best_params['mus'].numpy()
        best_sigmas = self.early_stop.best_params['sigmas'].numpy()

        self.assertNotEqual(list(best_mus), [3.0, 4.0], "best_params mus should not change with original params")
        self.assertNotEqual(list(best_sigmas), [0.3, 0.4], "best_params sigmas should not change with original params")

        # Ensure the values are still at their best recorded state
        np.testing.assert_array_almost_equal(best_mus, [1.0, 2.0], err_msg="best_params mus should retain best values")
        np.testing.assert_array_almost_equal(best_sigmas, [0.1, 0.2], err_msg="best_params sigmas should retain best values")

if __name__ == '__main__':
    unittest.main()
