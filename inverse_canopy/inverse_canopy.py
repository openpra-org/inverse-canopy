"""
main module for inverse_canopy, implementing the inverse UQ algorithm.
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import tf_keras
from tf_keras.constraints import NonNeg

from .lognormal_utils import compute_mu_sigma
from .early_stop import EarlyStop
from .metrics import summarize_predicted_end_states, summarize_predicted_conditional_events


def initialize_conditional_events(events, dtype, freeze_initiating_event=False):
    bounds = {
        "means": tf.constant([events['bounds']['mean']['min'], events['bounds']['mean']['max']], dtype=dtype),
        "stds": tf.constant([events['bounds']['std']['min'], events['bounds']['std']['max']], dtype=dtype)
    }

    num_events = len(events['names'])
    mask_mus = [1 for _ in range(0, num_events)]
    mask_sigmas = [1 for _ in range(0, num_events)]

    initial_means = [events['initial']['mean'] for _ in range(0, num_events)]
    initial_stds = [events['initial']['std'] for _ in range(0, num_events)]

    if freeze_initiating_event:
        initial_means[0] = 1  # always 1 since it scales the end-state freqs.
        initial_stds[0] = events['bounds']['std']['min']  # minimum possible
        # do not try to optimize the first conditional event
        mask_mus[0] = 0
        mask_sigmas[0] = 0

    initial = {
        "means": tf.constant(initial_means, dtype=dtype),
        "stds": tf.constant(initial_stds, dtype=dtype)
    }
    mask = {
        "mus": tf.constant(mask_mus, dtype=dtype),
        "sigmas": tf.constant(mask_sigmas, dtype=dtype)
    }
    return bounds, initial, mask


def initialize_end_states(end_states, num_samples, dtype, ie_freq=1):
    probabilities = tf.constant(np.array([details['probability'] / ie_freq for details in end_states.values()]), dtype=dtype)
    ones_vector = tf.ones([num_samples], dtype=dtype)
    end_state_pdf = probabilities[:, tf.newaxis] * ones_vector
    event_sequences = tf.constant(np.array([details['sequence'] for details in end_states.values()]), dtype=dtype)
    return end_state_pdf, event_sequences


class InverseCanopy(tf.Module):
    def __init__(self, conditional_events, end_states, tunable):
        super().__init__()

        self.dtype = tunable['dtype']
        self.epsilon = tf.constant(tunable['epsilon'], dtype=self.dtype)
        self.num_samples = tunable['num_samples']

        if 'initiating_event_frequency' in tunable:
            self.initiating_event_frequency = tunable['initiating_event_frequency']
        else:
            self.initiating_event_frequency = 1

        if 'freeze_initiating_event' in tunable:
            self.freeze_initiating_event = tunable['freeze_initiating_event']
        else:
            self.freeze_initiating_event = False

        use_float32 = True if self.dtype.name.title() == 'float32' else False
        tf_keras.backend.set_floatx(str.lower(self.dtype.name.title()))
        tf_keras.backend.set_epsilon(self.epsilon)
        tf.config.experimental.enable_tensor_float_32_execution(use_float32)
        tf.print(f"tunable initialized: dtype={self.dtype}, epsilon={self.epsilon}")

        # initialize conditional events related parameters
        self.conditional_events = conditional_events
        bounds, initial_guess, trainable_mask = initialize_conditional_events(self.conditional_events, self.dtype, freeze_initiating_event=self.freeze_initiating_event)
        self.num_conditional_events = len(self.conditional_events['names'])

        # initialize internal model params
        mus, sigmas = compute_mu_sigma(initial_guess['means'], initial_guess['stds'], dtype=self.dtype)
        self.params_mus = tf.Variable(mus, dtype=self.dtype)
        self.params_sigmas = tf.Variable(sigmas, dtype=self.dtype, constraint=NonNeg())

        # set the trainable boolean masks
        self.trainable_mask_mus = tf.constant(trainable_mask["mus"], dtype=self.dtype)
        self.trainable_mask_sigmas = tf.constant(trainable_mask["sigmas"], dtype=self.dtype)

        # clipping constraints on mean, std
        self.constraints_means = tf.constant(bounds["means"], dtype=self.dtype)
        self.constraints_stds = tf.constant(bounds["stds"], dtype=self.dtype)

        # initialize end states related parameters
        self.end_states = end_states
        self.num_end_states = len(self.end_states)

        # declare created methods
        self.compute_mu_sigma_from_sampled_distributions = self._create_compute_mu_sigma_from_sampled_distributions()

        target_pdf, target_sequences = initialize_end_states(self.end_states, self.num_samples, self.dtype, ie_freq=self.initiating_event_frequency)
        target_log_pdf, target_mus, target_sigmas = self.compute_mu_sigma_from_sampled_distributions(target_pdf)

        self.targets = {
            'pdf': target_pdf,
            'sequences': target_sequences,
            'log_pdf': target_log_pdf,
            'mus': target_mus,
            'sigmas': target_sigmas
        }

        self.mae_loss = self._create_mae_loss()
        self.normalized_relative_logarithmic_error = self._create_normalized_relative_logarithmic_error()
        self.normalized_relative_logarithmic_loss = self._create_normalized_relative_logarithmic_loss()
        self.predict_end_state_likelihoods = self._create_predict_end_state_likelihoods()
        self.compute_y = self._create_compute_y()
        self.sample_from_distribution = self._create_sample_from_distribution()
        self.compute_mu_sigma = self._create_compute_mu_sigma()
        self.compute_mean_std = self._create_compute_mean_std()
        self.clip_mu_sigma = self._create_clip_mu_sigma()

    def _create_sample_from_distribution(self):

        num_samples = self.num_samples
        dtype = self.dtype
        epsilon = self.epsilon

        p_low = tf.math.log(tf.cast(0.0, dtype=dtype) + epsilon)
        p_high = tf.math.log(tf.cast(1.0, dtype=dtype))

        #@tf.function(input_signature=input_signature)
        def sample_from_distribution(mus, sigmas):
            sigmas_with_epsilon = sigmas + epsilon
            truncated_dist = tfp.distributions.TruncatedNormal(loc=mus, scale=sigmas_with_epsilon, low=p_low,
                                                               high=p_high)
            return tf.exp(truncated_dist.sample(num_samples))

        return sample_from_distribution

    def _create_compute_y(self):
        y_sequences = self.targets['sequences']

        #@tf.function(input_signature=input_signature)
        def compute_y(sampled_vars):
            # Expand dimensions of `sampled_vars` for broadcasting
            expanded_sampled_vars = tf.expand_dims(sampled_vars, axis=1)  # Shape:[num_samples,1,num_conditional_events]
            occurrence_mask = tf.equal(y_sequences, 1)  # Event occurred
            non_occurrence_mask = tf.equal(y_sequences, 0)  # Event did not occur

            # Apply masks
            # For occurrence, use sampled_vars
            # For non-occurrence, use 1 - sampled_vars
            # For 'ignore' (-1), use 1 to have no effect
            sampled_occurrence = tf.where(occurrence_mask, expanded_sampled_vars, 1)
            sampled_non_occurrence = tf.where(non_occurrence_mask, 1 - expanded_sampled_vars, sampled_occurrence)

            # Compute the product across the conditional events dimension
            y_combined = tf.reduce_prod(sampled_non_occurrence, axis=-1)  # shape [11, 2]
            return tf.transpose(y_combined)  # shape [2, 11]

        return compute_y

    def _create_compute_mu_sigma_from_sampled_distributions(self):
        #@tf.function(input_signature=input_signature)
        def compute_mu_sigma_from_sampled_distributions(y_dists):
            log_y = tf.math.log(y_dists + 1e-30)
            mus = tf.math.reduce_mean(log_y, axis=1)
            sigmas = tf.math.reduce_std(log_y, axis=1)
            return log_y, mus, sigmas

        return compute_mu_sigma_from_sampled_distributions

    def _create_predict_end_state_likelihoods(self):

        #@tf.function(input_signature=input_signature)
        def predict_end_state_likelihoods(mus, sigmas):
            predicted_samples = self.sample_from_distribution(mus, sigmas)  # num_samples
            y_pred_pdf = self.compute_y(predicted_samples)  # y_sequences
            return y_pred_pdf

        return predict_end_state_likelihoods

    def _create_mae_loss(self):
        #@tf.function(input_signature=input_signature)
        def mae_loss(y_true, y_pred):
            loss = tf.reduce_mean(tf.abs(y_true - y_pred), axis=1)
            return loss

        return mae_loss

    def _create_normalized_relative_logarithmic_error(self):

        log_y_true = self.targets['log_pdf']
        sigma_y_true = self.targets['sigmas']
        #@tf.function(input_signature=input_signature)
        def normalized_relative_logarithmic_error(y_pred):
            log_y_pred, _, sigma_y_pred = self.compute_mu_sigma_from_sampled_distributions(y_pred)
            pdf_losses = self.mae_loss(log_y_true, log_y_pred)
            sigma_losses = tf.abs(sigma_y_true - sigma_y_pred)
            mean_stack = tf.reduce_mean(tf.stack([pdf_losses, sigma_losses]), axis=0)
            mean_mean_stack = tf.reduce_mean(mean_stack, axis=0)
            return mean_mean_stack

        return normalized_relative_logarithmic_error

    def _create_normalized_relative_logarithmic_loss(self):
        #@tf.function(input_signature=input_signature)
        def normalized_relative_logarithmic_loss(mus, sigmas):
            y_pred_pdf = self.predict_end_state_likelihoods(mus, sigmas)
            nrle = self.normalized_relative_logarithmic_error(y_pred_pdf)
            return nrle

        return normalized_relative_logarithmic_loss

    def _create_compute_mu_sigma(self):

        one = tf.cast(1.0, dtype=self.dtype)

        #@tf.function(input_signature=input_signature)
        def compute_mu_sigma(mean, std):
            mean_squared = tf.square(mean)
            std_squared = tf.square(std)
            mu = tf.math.log(mean_squared / tf.sqrt(std_squared + mean_squared))
            sigma = tf.sqrt(tf.math.log(one + std_squared / mean_squared))
            return mu, sigma

        return compute_mu_sigma

    def _create_compute_mean_std(self):
        one = tf.cast(1.0, dtype=self.dtype)
        two = tf.cast(2.0, dtype=self.dtype)
        #@tf.function(input_signature=input_signature)
        def compute_mean_std(mu, sigma):
            sigma_squared = tf.square(sigma)
            mean = tf.exp(mu + sigma_squared / two)
            std = tf.sqrt((tf.exp(sigma_squared) - one) * tf.exp(two * mu + sigma_squared))
            return mean, std

        return compute_mean_std

    def _create_clip_mu_sigma(self):
        constraints_mean = self.constraints_means
        constraints_std = self.constraints_stds
        dtype = self.dtype
        epsilon = self.epsilon
        neg_fifty = tf.cast(-50, dtype=dtype)
        six = tf.cast(6, dtype=dtype)

        #@tf.function(input_signature=input_signature)
        def clip_mu_sigma(mu, sigma):
            # Convert mu, sigma to mean, std and apply constraints
            mean, std = self.compute_mean_std(mu, sigma)
            mean = tf.clip_by_value(mean, constraints_mean[0], constraints_mean[1])
            std = tf.clip_by_value(std, constraints_std[0], constraints_std[1])

            # convert back to mu, sigma
            clipped_mu, clipped_sigma = self.compute_mu_sigma(mean, std)

            # clip for nonsensical values
            clipped_mu = tf.clip_by_value(clipped_mu, neg_fifty, -epsilon)
            clipped_sigma = tf.clip_by_value(clipped_sigma, epsilon, six)

            # ensure non-nans
            clipped_mu = tf.where(tf.math.is_nan(clipped_mu), tf.zeros_like(clipped_mu), clipped_mu)
            clipped_sigma = tf.where(tf.math.is_nan(clipped_sigma), tf.fill(clipped_sigma.shape, epsilon), clipped_sigma)
            return clipped_mu, clipped_sigma

        return clip_mu_sigma

    #@tf.function(jit_compile=True)
    def optimization_step(self, optimizer):
        with tf.GradientTape() as tape:
            loss = self.normalized_relative_logarithmic_loss(self.params_mus, self.params_sigmas)

        # Compute gradients with respect to model parameters that are variables
        gradients = tape.gradient(loss, [self.params_mus, self.params_sigmas])

        # # Prepare (gradient, variable) pairs, applying the mask to each gradient
        grads_and_vars = []

        # mu masks
        masked_grad_mus = gradients[0] * tf.cast(self.trainable_mask_mus, dtype=gradients[0].dtype)
        grads_and_vars.append((masked_grad_mus, self.params_mus))

        # sigma masks
        masked_grad_sigmas = gradients[1] * tf.cast(self.trainable_mask_sigmas, dtype=gradients[1].dtype)
        grads_and_vars.append((masked_grad_sigmas, self.params_sigmas))

        # Apply the masked gradients to the optimizer
        optimizer.apply_gradients(grads_and_vars)

        clipped_mu, clipped_sigma = self.clip_mu_sigma(self.params_mus, self.params_sigmas)
        self.params_mus.assign(clipped_mu)
        self.params_sigmas.assign(clipped_sigma)

        return loss

    """
    end-user helper function
    """
    def fit(self, learning_rate=0.1, patience=10, min_improvement=0.001, steps=1000, seed=372):
        tf.print(f"learning_rate: {learning_rate},"
                 f"patience: {patience},"
                 f"min_improvement: {min_improvement},"
                 f"max_steps: {steps},"
                 f"seed: {seed}")

        tf.random.set_seed(seed)
        np.random.seed(seed)
        early_stop = EarlyStop(min_delta=min_improvement, patience=patience, dtype=self.dtype, mus=self.params_mus, sigmas=self.params_sigmas)
        optimizer = tf_keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=False, epsilon=self.epsilon)
        return self.train_model(optimizer=optimizer, early_stop=early_stop, steps=steps)

    def train_model(self, optimizer, early_stop, steps=1000):
        interval = 100  # Number of steps between logging
        total_steps = tf.constant(steps, dtype=tf.int32)
        steps_done = tf.Variable(0, dtype=tf.int32)
        start_time = tf.Variable(0.0, dtype=tf.float64)
        last_time = tf.Variable(0.0, dtype=tf.float64)

        # Compiled function with XLA
        @tf.function(jit_compile=True)
        def optimization_steps(num_steps):
            last_loss = tf.constant(0.0, dtype=self.dtype)
            for _ in tf.range(num_steps):
                last_loss = self.optimization_step(optimizer)
            return last_loss

        # Training loop without XLA, but still in graph mode
        @tf.function(jit_compile=False)
        def train_loop():
            start_time.assign(tf.timestamp())
            last_time.assign(tf.timestamp())
            steps_done.assign(0)
            early_stop.best_loss.assign(float('inf'))
            early_stop.epochs_since_improvement.assign(0)
            early_stop.should_stop.assign(False)
            early_stop.step_at_best_loss.assign(-1)

            loss = tf.constant(float('inf'), dtype=self.dtype)

            while tf.logical_and(steps_done < total_steps,
                                 tf.logical_not(early_stop.should_stop)):
                steps_to_run = tf.minimum(interval, total_steps - steps_done)
                loss = optimization_steps(steps_to_run)
                steps_done.assign_add(steps_to_run)

                # Early stopping logic
                early_stop(current_loss=loss, step=steps_done, mus=self.params_mus, sigmas=self.params_sigmas)

                # Print statistics
                now = tf.timestamp()
                elapsed_time = now - last_time
                total_elapsed_time = now - start_time
                its_per_sec = tf.cast(steps_done, tf.float64) / total_elapsed_time
                interval_its_per_sec = tf.cast(steps_to_run, tf.float64) / elapsed_time
                tf.print("Step", steps_done, ": Loss =", loss,
                         ", Avg Iterations per second =", its_per_sec,
                         ", Interval Iterations per second =", interval_its_per_sec)
                last_time.assign(now)

        # Start the training loop
        train_loop()

    def summarize(self, show_plot=True, show_metrics=True, scale_by_initiating_event_frequency=True):
        summarize_predicted_end_states(self, show_plot=show_plot, show_metrics=show_metrics)
        if show_metrics:
            summarize_predicted_conditional_events(self)
