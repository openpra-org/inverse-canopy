import tensorflow as tf
import numpy as np
import unittest

from inverse_canopy import InverseCanopy

class TestInverseCanopy(unittest.TestCase):
    def setUp(self):

        self.tunable = {
            'num_samples': 10000,  # number of monte carlo samples
            'learning_rate': 0.005,  # the gradient update rate
            'dtype': tf.float64,  # use 64-bit floats
            'epsilon': 1e-20,  # useful for avoiding log(0 + epsilon) type errors
            'max_steps': 6000,  # maximum steps, regardless of convergence
            'patience': 4000,  # number of steps to wait before early stopping if the loss does not improve
            'initiating_event_frequency': 5.0e-1,
            'freeze_initiating_event': True,
        }

        self.conditional_events = {
            'names': ['LF1A', 'FSIG', 'FROD', 'PRUN|FROD', 'BPHR', 'DHRS', 'DHRL|~BPHR', 'DHRL|~DHRS'],

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
            },

            'LF1A': {
                'trainable': {
                    'mu': False,
                    'sigma': False,
                },
                '5th': {
                    'min': 5e-1,
                    'max': 5e-1,
                    'initial': 5e-1,
                },
                'mean': {
                    'min': 5e-1,
                    'max': 5e-1,
                    'initial': 5e-1,
                },
                '95th': {
                    'min': 5e-1,
                    'max': 5e-1,
                    'initial': 5e-1,
                },
            },

            'FSIG': {
                'trainable': {
                    'mu': True,
                    'sigma': True,
                },
                '5th': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 1e-18,
                },
                'mean': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 5e-1,
                },
                '95th': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 1e0,
                },
            },

            'FROD': {
                'trainable': {
                    'mu': True,
                    'sigma': True,
                },
                '5th': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 1e-18,
                },
                'mean': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 5e-1,
                },
                '95th': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 1e0,
                },
            },

            'PRUN|FROD': {
                'trainable': {
                    'mu': True,
                    'sigma': True,
                },
                '5th': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 1e-18,
                },
                'mean': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 5e-1,
                },
                '95th': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 1e0,
                },
            },

            'BPHR': {
                'trainable': {
                    'mu': True,
                    'sigma': True,
                },
                '5th': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 1e-18,
                },
                'mean': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 5e-1,
                },
                '95th': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 1e0,
                },
            },

            'DHRS': {
                'trainable': {
                    'mu': True,
                    'sigma': True,
                },
                '5th': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 1e-18,
                },
                'mean': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 5e-1,
                },
                '95th': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 1e0,
                },
            },

            'DHRL|~BPHR': {
                'trainable': {
                    'mu': True,
                    'sigma': True,
                },
                '5th': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 1e-18,
                },
                'mean': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 5e-1,
                },
                '95th': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 1e0,
                },
            },

            'DHRL|~DHRS': {
                'trainable': {
                    'mu': True,
                    'sigma': True,
                },
                '5th': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 1e-18,
                },
                'mean': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 5e-1,
                },
                '95th': {
                    'min': 1e-18,
                    'max': 1e0,
                    'initial': 1e0,
                },
            },
        }

        self.end_states = {
            'LF1A-1': {
                'sequence': [1, 0, 0, np.nan, 0, np.nan, 0, np.nan],
                'probability': 5.0e-1,
            },
            'LF1A-2': {
                'sequence': [1, 0, 0, np.nan, 0, np.nan, 1, np.nan],
                'probability': 4.20e-9,
            },
            'LF1A-3': {
                'sequence': [1, 0, 0, np.nan, 1, 0, np.nan, 0],
                'probability': 4.55e-4,
            },
            'LF1A-4': {
                'sequence': [1, 0, 0, np.nan, 1, 0, np.nan, 1],
                'probability': 7.4e-11,
            },
            'LF1A-5': {
                'sequence': [1, 0, 0, np.nan, 1, 1, np.nan, np.nan],
                'probability': 2.90e-10,
            },
            'LF1A-6': {
                'sequence': [1, 0, 1, 0, np.nan, np.nan, np.nan, np.nan],
                'probability': 2.30e-6,
            },
            'LF1A-7': {
                'sequence': [1, 0, 1, 1, np.nan, np.nan, np.nan, np.nan],
                'probability': 4.5e-9,
            },
            'LF1A-8': {
                'sequence': [1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                'probability': 8.9e-8,
            },
        }

    def test_simple_run(self):
        print('Devices: ', tf.config.list_physical_devices())
        model = InverseCanopy(self.conditional_events, self.end_states, self.tunable)
        model.fit(steps=self.tunable['max_steps'], patience=self.tunable['patience'], learning_rate=self.tunable['learning_rate'])
        model.summarize(show_plot=True, show_metrics=False)

if __name__ == '__main__':
    unittest.main()