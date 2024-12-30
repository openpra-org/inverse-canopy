from inverse_canopy import InverseCanopy
import tensorflow as tf
import numpy as np
import os
tunable = {
    'num_samples': 2,  # number of monte carlo samples
    'learning_rate': 0.1,  # the gradient update rate
    'dtype': tf.float64,  # use 64-bit floats
    'epsilon': 1e-30,  # useful for avoiding log(0 + epsilon) type errors
    'max_steps': 5000,  # maximum steps, regardless of convergence
    'patience': 50,  # number of steps to wait before early stopping if the loss does not improve
    'initiating_event_frequency': 5.0e-1,
    'freeze_initiating_event': True,
}

conditional_events = {
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

end_states = {
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


if __name__ == '__main__':
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    TF_XLA_FLAGS = "--tf_xla_enable_lazy_compilation=false --tf_xla_auto_jit=2"
    TF_XLA_FLAGS += " --tf_mlir_enable_mlir_bridge=true --tf_mlir_enable_convert_control_to_data_outputs_pass=true --tf_mlir_enable_multiple_local_cpu_devices=true"
    TF_XLA_FLAGS += " --tf_xla_deterministic_cluster_names=true --tf_xla_disable_strict_signature_checks=true"
    TF_XLA_FLAGS += " --tf_xla_persistent_cache_directory='./xla/cache/' --tf_xla_persistent_cache_read_only=false"

    #os.environ["TF_XLA_FLAGS"] = TF_XLA_FLAGS

    print('Devices: ', tf.config.list_physical_devices())
    model = InverseCanopy(conditional_events, end_states, tunable)
    model.fit(steps=tunable['max_steps'], patience=tunable['patience'], learning_rate=tunable['learning_rate'])
    model.summarize(show_plot=True, show_metrics=False)