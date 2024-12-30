import unittest
from inverse_canopy import EarlyStop, ModelParams


class TestEarlyStopBasicFunctionality(unittest.TestCase):
    def setUp(self):
        """Set up the EarlyStop object and a dummy ModelParams object for testing."""
        self.early_stop = EarlyStop()
        self.params = ModelParams()

    def test_loss_improvement(self):
        """Test the behavior when the loss improves sufficiently."""
        initial_loss = 1.0
        improved_loss = 0.98  # Improvement greater than min_delta (0.001)
        self.early_stop(initial_loss, step=1, params=self.params)
        self.early_stop(improved_loss, step=2, params=self.params)

        self.assertEqual(self.early_stop.best_loss, improved_loss, "Best loss should update to improved loss")
        self.assertEqual(self.early_stop.step_at_best_loss, 2, "Step at best loss should update")
        self.assertEqual(self.early_stop.epochs_since_improvement, 0, "Epochs since improvement should reset to 0")
        self.assertFalse(self.early_stop.should_stop, "Should not stop after improvement")

    def test_insufficient_loss_improvement(self):
        """Test the behavior when the loss does not improve sufficiently."""
        initial_loss = 1.0
        insufficient_improvement_loss = 0.999  # Improvement exactly equal to min_delta (0.001)
        self.early_stop(initial_loss, step=1, params=self.params)
        self.early_stop(insufficient_improvement_loss, step=2, params=self.params)

        self.assertEqual(self.early_stop.best_loss, insufficient_improvement_loss,
                         "Best loss should update to 0.999 since improvement equals min_delta")
        self.assertEqual(self.early_stop.step_at_best_loss, 2, "Step at best loss should not change")
        self.assertEqual(self.early_stop.epochs_since_improvement, 0, "Epochs since improvement should not increment")
        self.assertFalse(self.early_stop.should_stop, "Should not stop without sufficient improvement")

    def test_constant_loss(self):
        """Test the behavior when the loss remains constant over multiple steps."""
        constant_loss = 1.0
        for step in range(1, 5):
            self.early_stop(constant_loss, step=step, params=self.params)

        self.assertEqual(self.early_stop.best_loss, constant_loss, "Best loss should be the initial constant loss")
        self.assertEqual(self.early_stop.step_at_best_loss, 1, "Step at best loss should be the first step")
        self.assertEqual(self.early_stop.epochs_since_improvement, 3,
                         "Epochs since improvement should count all steps after the first")
        self.assertFalse(self.early_stop.should_stop,
                         "Should not stop if loss is constant and patience has not been exceeded")


if __name__ == '__main__':
    unittest.main()
