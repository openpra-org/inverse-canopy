"""

# Each key represents an end-state. Since the dictionary is ordered, the entires represent the order of the end-states.
# Within the end-state object are the `sequence`, `probability`, and `frequency` entries. `sequence` is mandatory.
# Between `probability` and `frequency`, exactly one must be provided. If both `probability` & `frequency` are provided,
# precedence will be given to `frequency` and a warning will be thrown.
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
"""
