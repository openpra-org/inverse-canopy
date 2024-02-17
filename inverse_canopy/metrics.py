import tensorflow as tf
import tensorflow_probability as tfp

from .plot import plot_predicted_end_states


def compute_metrics(pdf, axis=0):
    mean = tf.reduce_mean(pdf, axis=axis)
    p05 = tfp.stats.percentile(pdf, 5.0, axis=axis)
    p95 = tfp.stats.percentile(pdf, 95.0, axis=axis)
    return p05, mean, p95


def summarize_metrics(metrics, ids, metric_names=f"5th       Mean      95th"):
    initial_spaces = (max(len(s) for s in ids))
    metric_names = (' ' * (initial_spaces + 3)) + metric_names
    print(metric_names)
    for i, event in enumerate(ids):
        line = f"{event}" + (" " * (initial_spaces - len(event)))
        for metric in metrics:
            line += f"  {metric.numpy()[i]:.2e}"
        print(line)
    print("\n")
    pass


def summarize_predicted_conditional_events(model, scale_initiating_event_frequency=True):
    predicted_mus, predicted_sigmas = model.params.spread()
    predicted_samples = model.sample_from_distribution(predicted_mus, predicted_sigmas)
    p05, mean, p95 = compute_metrics(predicted_samples)
    conditional_events = model.conditional_events['names']
    initial_spaces = (max(len(s) for s in conditional_events) + 30)
    print(f"predicted conditional events")
    print("-" * initial_spaces)
    summarize_metrics([p05, mean, p95], conditional_events, metric_names=f"5th       Mean      95th")
    pass


def summarize_predicted_end_states(model, show_plot=True, show_metrics=True, scale_by_initiating_event_frequency=True):
    predicted_mus, predicted_sigmas = model.params.spread()
    y_pred_pdf = model.predict_end_state_likelihoods(predicted_mus, predicted_sigmas)
    y_obs_pdf = model.targets['pdf']

    if scale_by_initiating_event_frequency:
        y_pred_pdf *= model.initiating_event_frequency
        y_obs_pdf *= model.initiating_event_frequency

    end_state_names = [name for name in model.end_states.keys()]
    if show_metrics:
        p05, mean, p95 = compute_metrics(y_pred_pdf, axis=1)
        initial_spaces = (max(len(s) for s in end_state_names) + 30)
        print(f"predicted end states")
        print("-" * initial_spaces)
        summarize_metrics([p05, mean, p95], end_state_names, metric_names=f"5th       Mean      95th")

    if show_plot:
        plot_predicted_end_states(y_pred_pdf, y_obs_pdf, names=end_state_names)
    pass
