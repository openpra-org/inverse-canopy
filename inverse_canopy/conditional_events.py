"""
Initialize conditional events based on input parameters by performing bounds and validity checks.

# By default, all conditional_events are assumed to be trainable.
conditional_events = {
    # specifies the order of the events encoding vector
    'order': ['LF1A', 'FSIG', 'FROD', 'PRUN|FROD', 'BPHR', 'DHRS', 'DHRL|~BPHR', 'DHRL|~DHRS'],

    # global bounds for all conditional event parameters. Each field `mean`,`std`,`5th`,`95th`,`error_factor` & their
    # internals `min` and `max` are optional. When not provided, the defaults are assumed to be the values below. When
    # multiple fields are provided, the order of precedence is [std, 5th, 95th, error_factor], with `std` as the highest
    # precedence. For any field, if the min is larger than the max, a warning is thrown and the values are flipped.
    # Bounds will be checked for validity, for e.g., `mean`, `5th`, `95th`, cannot be outside [0,1]. Sometimes these
    # bounds need to be exceeded (for e.g., when dealing with frequencies). In this case, a warning will be thrown once
    # per bounds violation. `error_factor` cannot be less than 1. `std` is strictly greater than 0. No field can be NaN.
    # Note: since the model internal parameters are {mu, sigma} these values will be used to compute {mu, sigma}.
    'bounds': {
        'mean': {
            'min': 1e-14,
            'max': 1.00,
        },
        'std': {
            'min': 1e-10,
            'max': 1e8,
        },
        '5th': {
            'min': 1e-14,
            'max': 1.00,
        },
        '95th': {
            'min': 1e-14,
            'max': 1.00,
        },
        'error_factor': {
            'min': 1.00,
            'max': 1e8,
        }
     },

    # global initial values for all conditional events. Each field is optional. When not provided, the defaults are
    # assumed to be the values for `mean` and `std` below. When multiple fields are provided, the order of precedence is
    # [std, 5th, 95th, error_factor], with `std` as the highest precedence. Initials will be checked for validity, for
    # e.g., `mean`, `5th`, `95th`, cannot be outside [0,1]. Sometimes these bounds need to be exceeded (for e.g., when
    # dealing with frequencies). In this case, a warning will be thrown once per bounds violation. `error_factor`
    # cannot be less than 1. `std` is strictly greater than 0. No field can be a NaN.
    # Note: since the model internal parameters are `{mu, sigma}` these values will be used to compute `{mu, sigma}`.
    'initial': {
       'mean': 5e-1,
       'std': 1e8,
       '5th': 1e-14,
       '95th': 1e0,
       'error_factor': 1e8,
    },

    # global defaults are helpful, but local overrides are often needed. For example, we might want to set custom
    # constraints on the initiating event frequency, in this case the `LF1A` event. This can be achieved by providing
    # a `bounds`, `initial`, or `trainable` dictionary, depending on the need. In this example, since the initiating
    # event frequency is a known value with some uncertainty, we can fix it in place, ensuring that it's unmodifiable.
    # Here, both the `{mu, sigma}` parameters have be fixed and the `initial` value has been set. Note that the `bounds`
    # dictionary is not needed since the initial value will never change. As with the `bounds` & `initial` dictionaries
    # previously described, the same rules apply here.
    'LF1A': {
        'trainable': {
            'mu': False,
            'sigma': False,
        },
        'initial': {
           'mean': 1.5,
           'std': 0.2
        },
    },
}
"""

import tensorflow as tf


def parse_bounds(conditional_events, dtype=tf.float64):
    """
    stats_tensor: A TensorFlow tensor of shape (n, 3, 2) where n is the number of events,
      the second dimension represents the 5th percentile, mean, and 95th percentile,
      and the last dimension represents the min and max values.
    """
    # Initialize an empty list to store the stats for each event
    stats_list = []

    # Iterate through the events in the specified order
    for event in conditional_events['names']:
        event_data = conditional_events[event]
        # Extract the 5th percentile, mean, and 95th percentile stats
        stats = [
            [event_data['5th']['min'], event_data['5th']['max']],
            [event_data['mean']['min'], event_data['mean']['max']],
            [event_data['95th']['min'], event_data['95th']['max']],
        ]
        stats_list.append(stats)

    # Convert the list to a TensorFlow tensor
    stats_tensor = tf.constant(stats_list, dtype=dtype)
    return stats_tensor


def parse_initial_guess_tensor(conditional_events):
    """
    Creates a TensorFlow tensor of initial guess values for the 5th percentile, mean, and 95th percentile
    for each event in the conditional_events dictionary.

    Args:
        conditional_events (dict): A dictionary containing event data, including initial guess values.

    Returns:
        tf.Tensor: A TensorFlow tensor of shape (n, 3, 1) where n is the number of events,
                   the second dimension represents the 5th percentile, mean, and 95th percentile,
                   and the last dimension represents the initial guess.
    """
    # Initialize an empty list to store the initial guess stats for each event
    initial_guess_list = []

    # Iterate through the events in the specified order
    for event in conditional_events['order']:
        event_data = conditional_events[event]
        # Extract the initial guess for the 5th percentile, mean, and 95th percentile stats
        initial_guess = [
            [event_data['5th']['initial']],
            [event_data['mean']['initial']],
            [event_data['95th']['initial']],
        ]
        initial_guess_list.append(initial_guess)

    # Convert the list to a TensorFlow tensor
    initial_guess_tensor = tf.constant(initial_guess_list, dtype=tf.float64)

    return initial_guess_tensor
