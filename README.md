# InverseCanopy

Back-fit conditional/functional event probability distributions in an event tree to match target end-state
frequencies.

## How to Use

Below is a step-by-step example of how to use `inverse-canopy` in your project.

### Step 1: Import the Package

First, you need to import `InverseCanopy` from `inverse_canopy`, along with `tensorflow` and `numpy`.

```python
from inverse_canopy import InverseCanopy
import tensorflow as tf
import numpy as np
```

### Step 2: Define Your Parameters

You'll need to set up some parameters for the model to work with. These include the number of samples, learning rate, data type, and more.

```python
tunable = {
 'num_samples': 100,                    # Number of Monte Carlo samples, you don't need too many for smooth functions
 'learning_rate': 0.1,                  # Learning rate for gradient updates
 'dtype': tf.float64,                   # Use 64-bit floats for calculations
 'epsilon': 1e-30,                      # Helps avoid log(0) errors
 'max_steps': 5000,                     # Maximum optimization steps
 'patience': 50,                        # Steps to wait for improvement before stopping
 'initiating_event_frequency': 5.0e-1,  # set the initiating event (IE) frequency here
 'freeze_initiating_event': True,       # set to False if you'd like to predict the IE frequency as well
}
```

### Step 3: Set Up Conditional Events and End States

Define the conditional events and end states for your model. This includes names, bounds for mean and standard deviation, initial values, and the probabilities for each end state.

```python
conditional_events = {
    'names': ['OCSP', 'RSIG', 'RROD', 'SPTR', 'BPHR', 'DHRS|BPHR', 'DHRS|~SPTR', 'DHRL|~BPHR', 'DHRL|~DHRS|BPHR', 'DHRL|~DHRS|~SPTR'],
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

end_states = {
    'OCSP-1': {
        'sequence': [1, 0, 0, np.nan, 0, np.nan, np.nan, 0, np.nan, np.nan],
        'probability': 3.24e-2,
    },
    'OCSP-2': {
        'sequence': [1, 0, 0, np.nan, 0, np.nan, np.nan, 1, np.nan, np.nan],
        'probability': 2.80e-10,
    },
    'OCSP-3': {
        'sequence': [1, 0, 0, np.nan, 1, 0, np.nan, np.nan, 0, np.nan],
        'probability': 5.81e-4,
    },
    'OCSP-4': {
        'sequence': [1, 0, 0, np.nan, 1, 0, np.nan, np.nan, 1, np.nan],
        'probability': 1.0e-11,
    },
    'OCSP-5': {
        'sequence': [1, 0, 0, np.nan, 1, 1, np.nan, np.nan, np.nan, np.nan],
        'probability': 1.9e-11,
    },
    'OCSP-6': {
        'sequence': [1, 0, 1, 0, np.nan, np.nan, 0, np.nan, np.nan, 0],
        'probability': 1.0e-11,
    },
    'OCSP-7': {
        'sequence': [1, 0, 1, 0, np.nan, np.nan, 0, np.nan, np.nan, 1],
        'probability': 1.0e-11,
    },
    'OCSP-8': {
        'sequence': [1, 0, 1, 0, np.nan, np.nan, 1, np.nan, np.nan, np.nan],
        'probability': 1.0e-11,
    },

    'OCSP-9': {
        'sequence': [1, 0, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'probability': 1.5e-10,
    },

    'OCSP-10': {
        'sequence': [1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'probability': 1.10e-7,
    },    
}
```

### Step 4: Initialize and Fit the Model

Create an instance of `InverseCanopy` with your conditional events, end states, and tunable parameters. Then, fit the model.

```python
model = InverseCanopy(conditional_events, end_states, tunable)
model.fit(steps=tunable['max_steps'], patience=tunable['patience'], learning_rate=tunable['learning_rate'])
```

### Step 5: Summarize the Results

Finally, you can summarize the results of the model. This can include showing a plot of the results and displaying metrics.

```python
model.summarize(show_plot=True, show_metrics=True)
```

And that's it! You've successfully used `inverse-canopy` to back-fit conditional/functional event probabilities in an 
event tree.


## Example Output
Checkout the [demo jupyter notebook](notebooks/demo.ipynb).

```jupyterpython
model = InverseCanopy(conditional_events, end_states, tunable)
model.fit(steps=tunable['max_steps'], patience=tunable['patience'], learning_rate=tunable['learning_rate'])
```
```pycon
tunable initialized: dtype=<dtype: 'float64'>, epsilon=1e-20
learning_rate: 0.1,patience: 50,min_improvement: 0.001,max_steps: 5000,seed: 372
Step 0: Loss = 16.2806382972511976, performing 204.6 it/sec
Step 100: Loss = 2.2360882617814344, performing 923.6 it/sec
Step 200: Loss = 0.8437262702306174, performing 1008.8 it/sec
Step 300: Loss = 0.5984919101511645, performing 985.0 it/sec
Step 400: Loss = 0.3233056051040787, performing 982.3 it/sec
Step 500: Loss = 0.0791769886078191, performing 951.4 it/sec
Step 600: Loss = 0.0563040994301810, performing 983.0 it/sec
Step 700: Loss = 0.0156518763008722, performing 983.6 it/sec
No improvement since Step 669, early stopping.
[Best]  Step 668: Loss = 0.0109982481161411
[Final] Step 718: Loss = 0.0117771872908507

predicted end states
-------------------------------------
          5th       Mean      95th
OCSP-0   9.67e-01  9.67e-01  9.67e-01
OCSP-1   3.24e-02  3.24e-02  3.24e-02
OCSP-2   2.73e-10  2.78e-10  2.84e-10
OCSP-3   5.80e-04  5.80e-04  5.80e-04
OCSP-4   1.02e-11  1.03e-11  1.04e-11
OCSP-5   1.89e-11  1.90e-11  1.90e-11
OCSP-6   9.38e-12  9.68e-12  1.00e-11
OCSP-7   9.36e-12  9.66e-12  9.97e-12
OCSP-8   9.36e-12  9.66e-12  9.98e-12
OCSP-9   1.46e-10  1.51e-10  1.55e-10
OCSP-10  1.08e-07  1.08e-07  1.08e-07


predicted conditional events
----------------------------------------------
                   5th       Mean      95th
OCSP              3.30e-02  3.30e-02  3.30e-02
RSIG              3.28e-06  3.28e-06  3.28e-06
RROD              5.27e-09  5.44e-09  5.62e-09
SPTR              8.38e-01  8.38e-01  8.38e-01
BPHR              1.76e-02  1.76e-02  1.76e-02
DHRS|BPHR         3.25e-08  3.27e-08  3.28e-08
DHRS|~SPTR        3.33e-01  3.33e-01  3.33e-01
DHRL|~BPHR        8.43e-09  8.59e-09  8.77e-09
DHRL|~DHRS|BPHR   1.76e-08  1.78e-08  1.79e-08
DHRL|~DHRS|~SPTR  4.99e-01  4.99e-01  4.99e-01
```

```jupyterpython
model.summarize(show_plot=True, show_metrics=False)
```
![output](https://gcdnb.pbrd.co/images/DDj4Nv536IJh.png?o=1 "demo output plot")

## Upload to PyPI

- `python setup.py sdist bdist_wheel`
- `pip install twine`
- `twine upload dist/*`