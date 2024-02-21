"""
Module: metrics

This module contains functions for computing and summarizing metrics for probability distribution functions and
predicted events based on a model's parameters.

Functions:
- compute_metrics: Compute the 5th, mean, and 95th percentile of a probability distribution along a specified axis.
- summarize_metrics: Summarize the computed metrics for each event.
- summarize_predicted_conditional_events: Summarize the predicted conditional events based on the model's parameters.
- summarize_predicted_end_states: Summarize the predicted end states based on the model's parameters, optionally display
                                  a plot.

Dependencies:
- tensorflow
- tensorflow_probability
- inverse_canopy.plot
"""

import tensorflow as tf
import tensorflow_probability as tfp
from .plot import plot_predicted_end_states


def compute_metrics(pdf, axis=0):
    """
    Compute the 5th percentile, mean, and 95th percentile of a probability distribution function along a specified axis.

    Args:
        pdf (tf.Tensor): The probability distribution function.
        axis (int): The axis along which to compute the metrics.

    Returns:
        tuple: A tuple containing the 5th percentile, mean, and 95th percentile.
    """
    mean = tf.reduce_mean(pdf, axis=axis)
    p05 = tfp.stats.percentile(pdf, 5.0, axis=axis)
    p95 = tfp.stats.percentile(pdf, 95.0, axis=axis)
    return p05, mean, p95


def summarize_metrics(metrics, ids, metric_names="5th       Mean      95th"):
    """
    Summarize the computed metrics for each event.

    Args:
        metrics (list): List of metrics to summarize.
        ids (list): List of event IDs.
        metric_names (str): Names of the metrics to display.

    Returns:
        None
    """
    initial_spaces = (max(len(s) for s in ids))
    metric_names = (' ' * (initial_spaces + 3)) + metric_names
    print(metric_names)
    for i, event in enumerate(ids):
        line = f"{event}" + (" " * (initial_spaces - len(event)))
        for metric in metrics:
            line += f"  {metric.numpy()[i]:.2e}"
        print(line)
    print("\n")


def summarize_predicted_conditional_events(model, scale_initiating_event_frequency=True):
    """
    Summarize the predicted conditional events based on the model's parameters.

    Args:
        model: The model object containing the predicted parameters.
        scale_initiating_event_frequency (bool): Whether to scale by the initiating event frequency.

    Returns:
        None
    """
    predicted_mus, predicted_sigmas = model.params.spread()
    predicted_samples = model.sample_from_distribution(predicted_mus, predicted_sigmas)
    p05, mean, p95 = compute_metrics(predicted_samples)
    conditional_events = model.conditional_events['names']
    initial_spaces = max(len(s) for s in conditional_events) + 30
    print("predicted conditional events")
    print("-" * initial_spaces)
    summarize_metrics([p05, mean, p95], conditional_events, metric_names="5th       Mean      95th")


def summarize_predicted_end_states(model, show_plot=True, show_metrics=True, scale_by_initiating_event_frequency=True):
    """
    Summarize the predicted end states based on the model's parameters and optionally display a plot.

    Args:
        model: The model object containing the predicted parameters.
        show_plot (bool): Whether to display a plot of the predicted end states.
        show_metrics (bool): Whether to display summary metrics for the predicted end states.
        scale_by_initiating_event_frequency (bool): Whether to scale by the initiating event frequency.

    Returns:
        None
    """
    predicted_mus, predicted_sigmas = model.params.spread()
    y_pred_pdf = model.predict_end_state_likelihoods(predicted_mus, predicted_sigmas)
    y_obs_pdf = model.targets['pdf']

    if scale_by_initiating_event_frequency:
        y_pred_pdf *= model.initiating_event_frequency
        y_obs_pdf *= model.initiating_event_frequency

    end_state_names = list(model.end_states.keys())
    if show_metrics:
        p05, mean, p95 = compute_metrics(y_pred_pdf, axis=1)
        initial_spaces = max(len(s) for s in end_state_names) + 30
        print("predicted end states")
        print("-" * initial_spaces)
        summarize_metrics([p05, mean, p95], end_state_names, metric_names="5th       Mean      95th")

    if show_plot:
        plot_predicted_end_states(y_pred_pdf, y_obs_pdf, names=end_state_names)
