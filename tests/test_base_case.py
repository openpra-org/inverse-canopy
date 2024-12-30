import unittest
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from inverse_canopy import InverseCanopy

class TestInverseCanopy(unittest.TestCase):
    def setUp(self):
        self.conditional_events = {
            'names': ['OCSP', 'RSIG', 'RROD', 'SPTR', 'BPHR', 'DHRS|BPHR', 'DHRS|~SPTR', 'DHRL|~BPHR', 'DHRL|~DHRS|BPHR', 'DHRL|~DHRS|~SPTR'],
            'bounds': {
                'mean': {'min': 1e-14, 'max': 1.00},
                'std': {'min': 1e-10, 'max': 1e8},
            },
            'initial': {'mean': 5e-1, 'std': 1e8}
        }
        self.end_states = {
            'OCSP-1': {'sequence': [1, 0, 0, np.nan, 0, np.nan, np.nan, 0, np.nan, np.nan], 'probability': 3.24e-2},
            # Additional end states would be defined here...
        }
        self.tunable = {
            'num_samples': 1000000,
            'learning_rate': 0.1,
            'dtype': tf.float64,
            'epsilon': 1e-30,
            'max_steps': 5000,
            'patience': 50,
            'initiating_event_frequency': 5.0e-1,
            'freeze_initiating_event': True,
        }
        self.model = InverseCanopy(self.conditional_events, self.end_states, self.tunable)

    def test_initialization(self):
        """Test the initialization of the InverseCanopy model."""
        self.assertIsInstance(self.model, InverseCanopy)

    @patch('inverse_canopy.InverseCanopy.fit')
    def test_fit_called(self, mock_fit):
        """Test that the fit method is called with correct parameters."""
        self.model.fit(steps=self.tunable['max_steps'], patience=self.tunable['patience'], learning_rate=self.tunable['learning_rate'], legacy=False)
        mock_fit.assert_called_once_with(steps=5000, patience=50, learning_rate=0.1, legacy=False)

    @patch('inverse_canopy.InverseCanopy.summarize')
    def test_summarize_called(self, mock_summarize):
        """Test that the summarize method is called correctly."""
        self.model.summarize(show_plot=True, show_metrics=True)
        mock_summarize.assert_called_once_with(show_plot=True, show_metrics=True)

    def test_parameters_set_correctly(self):
        """Test that parameters are set correctly in the model."""
        self.assertEqual(self.model.tunable['num_samples'], 1000000)
        self.assertEqual(self.model.tunable['learning_rate'], 0.1)
        self.assertEqual(self.model.tunable['dtype'], tf.float64)

    def test_conditional_events_structure(self):
        """Test the structure of conditional events."""
        self.assertTrue('names' in self.model.conditional_events)
        self.assertTrue('bounds' in self.model.conditional_events)
        self.assertTrue('initial' in self.model.conditional_events)

if __name__ == '__main__':
    unittest.main()