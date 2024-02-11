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
 'num_samples': 100,    # Number of Monte Carlo samples
 'learning_rate': 0.1,    # Learning rate for gradient updates
 'dtype': tf.float64,     # Use 64-bit floats for calculations
 'epsilon': 1e-30,        # Helps avoid log(0) errors
 'max_steps': 5000,       # Maximum optimization steps
 'patience': 50,          # Steps to wait for improvement before stopping   
}
```

### Step 3: Set Up Conditional Events and End States

Define the conditional events and end states for your model. This includes names, bounds for mean and standard deviation, initial values, and the probabilities for each end state.

```python
conditional_events = {
    'names': ['LCDL', 'PKRU', 'DHRS', 'DHRL'],
    'bounds': {
        'mean': {'min': 1e-14, 'max': 1.00},
        'std': {'min': 1e-10, 'max': 1e8},
    },
    'initial': {
       'mean': 5e-1,
       'std': 1e8,
    }
}

end_states = {
    'LCDL-1': {'sequence': [1, 0, 0, 0], 'probability': 1.1e-8},
    'LCDL-2': {'sequence': [1, 0, 0, 1], 'probability': 1.00e-11},
    'LCDL-3': {'sequence': [1, 0, 1, np.nan], 'probability': 1.00e-11},
    'LCDL-4': {'sequence': [1, 1, np.nan, np.nan], 'probability': 1.00e-11},
    'LCDL-0': {'sequence': [0, np.nan, np.nan, np.nan], 'probability': 1.0 - sum of other probabilities},
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
model.summarize(show_plot=True, show_metrics=False)
```
```pycon
tunable initialized: dtype=<dtype: 'float64'>, epsilon=1e-30
learning_rate: 0.1,patience: 50,min_improvement: 0.001,max_steps: 5000,seed: 372
Step 0: Loss = 17.8429693423585043
Step 100: Loss = 3.9290103273116728
Step 200: Loss = 0.0820853139891900
No improvement since Step 241, early stopping.
[Best]  Step 240: Loss = 0.0046576258420071
[Final] Step 290: Loss = 0.0112726303466675

predicted end states
-------------------------------------
         5th       Mean      95th
SDFR-0:  9.96e-01  9.96e-01  9.96e-01
SDFR-1:  4.23e-03  4.23e-03  4.23e-03
SDFR-2:  4.38e-05  4.38e-05  4.38e-05
SDFR-3:  6.24e-09  6.24e-09  6.24e-09
SDFR-4:  2.13e-05  2.13e-05  2.13e-05
SDFR-5:  1.91e-05  1.91e-05  1.91e-05
SDFR-6:  2.09e-06  2.09e-06  2.09e-06
SDFR-7:  1.74e-08  1.74e-08  1.74e-08
SDFR-8:  4.31e-09  4.31e-09  4.31e-09
-------------------------------------
predicted conditional events
-----------------------------------------
             5th       Mean      95th
SDFR      :  4.32e-03  4.32e-03  4.32e-03
LMFD      :  9.97e-07  9.98e-07  9.98e-07
RFIR      :  4.02e-06  4.02e-06  4.02e-06
LLRF      :  9.84e-03  9.84e-03  9.84e-03
SSSD|~LLRF:  1.02e-02  1.02e-02  1.02e-02
SSSD|LLRF :  4.98e-01  4.98e-01  4.98e-01
SYSO|~LLRF:  1.42e-04  1.42e-04  1.42e-04
SYSO|LLRF :  9.87e-02  9.87e-02  9.87e-02
-----------------------------------------
```

## Upload to PyPI

- `python setup.py sdist bdist_wheel`
- `pip install twine`
- `twine upload dist/*`