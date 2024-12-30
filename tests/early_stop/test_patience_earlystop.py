import unittest
from inverse_canopy import EarlyStop, ModelParams

class TestEarlyStopPatienceHandling(unittest.TestCase):
    def setUp(self):
        """Set up the EarlyStop object with a fixed min_delta and patience for testing."""
        self.min_delta = 0.001
        self.patience = 5
        self.early_stop = EarlyStop(min_delta=self.min_delta, patience=self.patience)
        self.initial_loss = 1.0
        self.model_params = ModelParams()  # Assuming ModelParams has a suitable constructor

    def test_patience_not_exceeded(self):
        """Test that should_stop remains False when patience is not yet exceeded."""
        for i in range(self.patience - 1):
            self.early_stop(self.initial_loss, i, self.model_params)
            self.assertFalse(self.early_stop.should_stop, "should_stop should be False when patience has not been exceeded")

    def test_patience_exceeded(self):
        """Test that should_stop becomes True when patience is exactly reached."""
        for i in range(self.patience):
            self.early_stop(self.initial_loss, i, self.model_params)
        self.assertTrue(self.early_stop.should_stop, "should_stop should be True when patience is exactly reached")

    def test_patience_exceeded_with_no_improvement(self):
        """Test should_stop behavior when there is no improvement over the patience limit."""
        for i in range(self.patience + 1):  # Go one step beyond the patience
            self.early_stop(self.initial_loss, i, self.model_params)
        self.assertTrue(self.early_stop.should_stop, "should_stop should be True when no improvement and patience is exceeded")

    def test_reset_patience_after_improvement(self):
        """Test the reset of patience after an improvement in loss."""
        for i in range(self.patience - 1):
            self.early_stop(self.initial_loss, i, self.model_params)
        # Introduce an improvement
        improved_loss = self.initial_loss - 10 * self.min_delta  # A clear improvement
        self.early_stop(improved_loss, self.patience - 1, self.model_params)
        # Continue with no improvement to check reset of patience
        for i in range(self.patience - 1):
            self.early_stop(improved_loss, self.patience + i, self.model_params)
            self.assertFalse(self.early_stop.should_stop, "should_stop should be False after improvement resets the patience")

if __name__ == '__main__':
    unittest.main()
