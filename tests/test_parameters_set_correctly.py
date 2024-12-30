import unittest
from inverse_canopy import InverseCanopy
import tensorflow as tf
import numpy as np

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
            'OCSP-1': {'sequence': [1, 0, 0, tf.constant(np.nan, dtype=tf.float64), 0, tf.constant(np.nan, dtype=tf.float64), tf.constant(np.nan, dtype=tf.float64), 0, tf.constant(np.nan, dtype=tf.float64), tf.constant(np.nan, dtype=tf.float64)], 'probability': 3.24e-2},
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

    def test_parameters_set_correctly(self):
        """Test that parameters are set correctly in the model."""
        self.assertEqual(self.model.num_samples, 1000000)
        self.assertEqual(self.model.dtype, tf.float64)
        self.assertTrue(tf.reduce_all(tf.equal(self.model.epsilon, tf.constant(1e-30, dtype=tf.float64))))
        self.assertEqual(self.model.initiating_event_frequency, 5.0e-1)
        self.assertTrue(self.model.freeze_initiating_event)

        # Check if the bounds and initial guesses are set correctly
        expected_means_bounds = tf.constant([1e-14, 1.00], dtype=tf.float64)
        expected_stds_bounds = tf.constant([1e-10, 1e8], dtype=tf.float64)
        self.assertTrue(tf.reduce_all(tf.equal(self.model.constraints['means'], expected_means_bounds)))
        self.assertTrue(tf.reduce_all(tf.equal(self.model.constraints['stds'], expected_stds_bounds)))

        # Check initial means and stds
        expected_initial_means = tf.constant([1] + [5e-1] * (len(self.conditional_events['names']) - 1), dtype=tf.float64)
        expected_initial_stds = tf.constant([1e-10] + [1e8] * (len(self.conditional_events['names']) - 1), dtype=tf.float64)
        self.assertTrue(tf.reduce_all(tf.equal(self.model.params['mus'], expected_initial_means)))
        self.assertTrue(tf.reduce_all(tf.equal(self.model.params['sigmas'], expected_initial_stds)))

if __name__ == '__main__':
    unittest.main()