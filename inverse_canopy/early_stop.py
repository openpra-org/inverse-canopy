"""
Module for implementing early stopping during training.
"""
import numpy as np
from .params import ModelParams

import tensorflow as tf


class EarlyStop(tf.Module):
    """
    Class for implementing early stopping during training.
    """

    def __init__(self, min_delta=0.001, patience=10, dtype=tf.float64):
        """
        Initialize the EarlyStop object with specified parameters.

        Args:
        min_delta (float): Minimum change in loss to be considered an improvement.
        patience (int): Number of epochs to wait for improvement before stopping.
        dtype: TensorFlow data type for variables.
        """
        super().__init__()
        self.min_delta = tf.constant(min_delta, dtype=dtype)
        self.patience = tf.constant(patience, dtype=tf.int32)
        self.best_loss = tf.Variable(float('inf'), dtype=dtype, trainable=False)
        self.epochs_since_improvement = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.should_stop = tf.Variable(False, dtype=tf.bool, trainable=False)
        self.step_at_best_loss = tf.Variable(-1, dtype=tf.int32, trainable=False)
        # Note: Handling self.best_params requires additional considerations.

    def __call__(self, current_loss, step):
        """
        Call method to update the early stopping criteria based on current loss.

        Args:
        current_loss: Current loss value (tf.Tensor)
        step: Current training step (tf.Tensor)
        """
        condition = tf.logical_or(
            tf.equal(self.best_loss, float('inf')),
            tf.greater(self.best_loss - current_loss, self.min_delta)
        )

        def update_best():
            self.best_loss.assign(current_loss)
            self.epochs_since_improvement.assign(0)
            self.step_at_best_loss.assign(step)
            # Note: Assigning the best parameters would require handling tf.Variables for parameters.
            return tf.constant(0)  # Dummy return

        def increment_epochs():
            self.epochs_since_improvement.assign_add(1)
            return tf.constant(0)  # Dummy return

        tf.cond(condition, update_best, increment_epochs)

        self.should_stop.assign(tf.greater_equal(self.epochs_since_improvement, self.patience))
