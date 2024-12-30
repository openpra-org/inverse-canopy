import unittest
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from inverse_canopy import InverseCanopy


class TestFixtureANTC(unittest.TestCase):
    def setUp(self):
        self.tunable = {
            'num_samples': 100,  # number of monte carlo samples
            'learning_rate': 0.05,  # the gradient update rate
            'dtype': tf.float64,  # use 64-bit floats
            'epsilon': 1e-20,  # useful for avoiding log(0 + epsilon) type errors
            'max_steps': 5000,  # maximum steps, regardless of convergence
            'patience': 50,  # number of steps to wait before early stopping if the loss does not improve
            'initiating_event_frequency': 5e-1,  # set the initiating event (IE) frequency here
            'freeze_initiating_event': True,  # set to False if you'd like to predict the IE frequency as well
        }

        self.conditional_events = {
            'names': ['ANTC', 'BPHR', 'DHRS'],
            'bounds': {
                'mean': {
                    'min': 1e-14,
                    'max': 1.00,
                },
                'std': {
                    'min': 1e-10,
                    'max': 1e8,
                },
            },
            'initial': {
                'mean': 5e-1,
                'std': 1e8,
            }
        }

        self.end_states = {
            'SHDL-1': {
                'sequence': [1, 0, np.nan],
                'probability': 5e-1,
            },
            'SHDL-2': {
                'sequence': [1, 1, 0],
                'probability': 1.57e-9,
            },
            'SHDL-3': {
                'sequence': [1, 1, 1],
                'probability': 2.9e-10,
            },

        }

    def test_fit(self):
        print('Devices: ', tf.config.list_physical_devices())
        model = InverseCanopy(self.conditional_events, self.end_states, self.tunable)
        model.fit(steps=self.tunable['max_steps'], patience=self.tunable['patience'], learning_rate=self.tunable['learning_rate'])
        model.summarize(show_plot=False, show_metrics=False)

if __name__ == '__main__':
    unittest.main()