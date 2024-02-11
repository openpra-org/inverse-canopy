import numpy as np


class EarlyStop:
    def __init__(self, min_delta=0.001, patience=10):
        self.min_delta = min_delta
        self.patience = patience
        self.best_loss = None
        self.epochs_since_improvement = 0
        self.should_stop = False
        self.step_at_best_loss = np.nan
        self.best_params = None

    def __call__(self, current_loss, step, params):
        if self.best_loss is None or (self.best_loss - current_loss) > self.min_delta:
            self.best_loss = current_loss
            self.epochs_since_improvement = 0
            self.step_at_best_loss = step
            self.best_params = params
        else:
            self.epochs_since_improvement += 1

        if self.epochs_since_improvement >= self.patience:
            self.should_stop = True
