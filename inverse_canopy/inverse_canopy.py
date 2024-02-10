import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.constraints import NonNeg
from .params import ModelInputs, TrainableParams, ModelParams


def initialize_conditional_events(events, dtype):
    bounds: ModelInputs = ModelInputs()
    bounds['means'] = tf.constant([events['bounds']['mean']['min'], events['bounds']['mean']['max']], dtype=dtype)
    bounds['stds'] = tf.constant([events['bounds']['std']['min'], events['bounds']['std']['max']], dtype=dtype)

    num_events = len(events['names'])
    initial: ModelInputs = ModelInputs()
    initial['means'] = tf.constant([events['initial']['mean'] for _ in range(0, num_events)], dtype=dtype)
    initial['stds'] = tf.constant([events['initial']['std'] for _ in range(0, num_events)], dtype=dtype)

    mask: TrainableParams = TrainableParams()
    mask['mus'] = tf.constant([1 for _ in range(0, num_events)], dtype=dtype)
    mask['sigmas'] = tf.constant([1 for _ in range(0, num_events)], dtype=dtype)

    return bounds, initial, mask


def initialize_end_states(end_states, num_samples, dtype):
    probabilities = tf.constant(np.array([details['probability'] for details in end_states.values()]), dtype=dtype)
    assert np.isclose(probabilities.numpy().sum(), 1.0, atol=1e-8), "Probabilities must sum to 1."
    end_state_pdf = tf.tile(tf.reshape(probabilities, (probabilities.shape[0], 1)), [1, num_samples])
    event_sequences = tf.constant(np.array([details['sequence'] for details in end_states.values()]), dtype=dtype)
    return end_state_pdf, event_sequences


class InverseCanopy(tf.Module):
    def __init__(self, conditional_events, end_states, tunable):

        # , initial_guess: ModelInputs, trainable_mask: TrainableParams, constraints: ModelInputs,
        #          dtype=tf.float64, epsilon=1e-30, num_samples=100):
        super().__init__()

        self.dtype = tunable['dtype']
        self.epsilon = tf.constant(tunable['epsilon'], dtype=self.dtype)
        self.num_samples = tunable['num_samples']

        tf.keras.backend.set_floatx(str.lower(self.dtype.name.title()))
        tf.keras.backend.set_epsilon(self.epsilon)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.print(f"tunables initialized: dtype={self.dtype}, epsilon={self.epsilon}")

        # initialize conditional events related parameters
        self.conditional_events = conditional_events
        bounds, initial_guess, trainable_mask = initialize_conditional_events(self.conditional_events, self.dtype)
        self.num_conditional_events = len(self.conditional_events['names'])

        # initialize internal model params
        self.params: ModelParams = initial_guess.to_musigma()
        self.params['mus'] = tf.Variable(self.params['mus'], dtype=self.dtype)
        self.params['sigmas'] = tf.Variable(self.params['sigmas'], dtype=self.dtype, constraint=NonNeg())

        # set the trainable boolean masks
        self.trainable: TrainableParams = trainable_mask
        self.trainable['mus'] = tf.constant(self.trainable['mus'], dtype=self.dtype)
        self.trainable['sigmas'] = tf.constant(self.trainable['sigmas'], dtype=self.dtype)

        # clipping constraints on mean, std
        self.constraints = bounds
        self.constraints['means'] = tf.constant(self.constraints['means'], dtype=self.dtype)
        self.constraints['stds'] = tf.constant(self.constraints['stds'], dtype=self.dtype)

        tf.print('params: ', self.params['mus'], self.params['sigmas'])
        tf.print('mask', self.trainable['mus'], self.trainable['sigmas'])
        tf.print('constraints', self.constraints['means'], self.constraints['stds'])

        # initialize end states related parameters
        self.end_states = end_states
        self.num_end_states = len(self.end_states)

        # declare created methods
        self.compute_mu_sigma_from_sampled_distributions = self._create_compute_mu_sigma_from_sampled_distributions()

        target_pdf, target_sequences = initialize_end_states(self.end_states, self.num_samples, self.dtype)
        target_log_pdf, target_mus, target_sigmas = self.compute_mu_sigma_from_sampled_distributions(target_pdf)

        self.targets = {
            'pdf': target_pdf,
            'sequences': target_sequences,
            'log_pdf': target_log_pdf,
            'mus': target_mus,
            'sigmas': target_sigmas
        }

        #tf.print(self.targets['pdf'].shape)
        self.mae_loss = self._create_mae_loss()
        self.normalized_relative_logarithmic_error = self._create_normalized_relative_logarithmic_error()
        self.normalized_relative_logarithmic_loss = self._create_normalized_relative_logarithmic_loss()
        self.predict_end_state_likelihoods = self._create_predict_end_state_likelihoods()
        self.compute_y = self._create_compute_y()
        self.sample_from_distribution = self._create_sample_from_distribution()

    def _create_sample_from_distribution(self):

        num_samples = self.num_samples
        dtype = self.dtype
        epsilon = self.epsilon

        # input_signature = [
        #     tf.TensorSpec(shape=[1, 2], dtype=self.dtype),
        # ]

        input_signature = [
            tf.TensorSpec(shape=(self.num_conditional_events, ), dtype=self.dtype),
            tf.TensorSpec(shape=(self.num_conditional_events, ), dtype=self.dtype),
        ]

        p_low = tf.math.log(tf.cast(0.0, dtype=dtype) + epsilon)
        p_high = tf.math.log(tf.cast(1.0, dtype=dtype))

        @tf.function(input_signature=input_signature)
        def sample_from_distribution(mus, sigmas):
            def sample_single(mu_sigma):
                #tf.print("loc:", mu_sigma[0], "scale:", mu_sigma[1], "low:", p_low, "high:", p_high)
                truncated = tfp.distributions.TruncatedNormal(loc=mu_sigma[0], scale=(mu_sigma[1] + epsilon),
                                                              low=p_low, high=p_high)
                return tf.exp(truncated.sample(num_samples))

            # Stack 'mus' and 'sigmas' along a new axis to create pairs
            mus_sigmas = tf.stack([mus, sigmas], axis=1)

            # Use 'tf.map_fn' to apply 'sample_single' across all 'mu_sigma' pairs
            samples = tf.map_fn(sample_single, mus_sigmas, fn_output_signature=dtype)

            # Transpose the samples to adjust the shape if necessary
            return tf.transpose(samples)

        return sample_from_distribution

    @staticmethod
    @tf.function
    def SequenceTF(sampled_vars, states_tensor):
        # Create a mask for non-NaN values (events to include)
        mask = tf.math.logical_not(tf.math.is_nan(states_tensor))

        # Replace NaN in states_tensor with 1 to have no effect when multiplying
        states_tensor = tf.where(mask, states_tensor, tf.ones_like(states_tensor))

        # Compute the product for each sample
        # For occurrence (1), use the sampled value; for non-occurrence (0), use 1 - the sampled value
        combination_samples = tf.reduce_prod(
            tf.where(tf.cast(mask, tf.bool),
                     tf.where(tf.cast(states_tensor, tf.bool), sampled_vars, 1 - sampled_vars),
                     tf.ones_like(sampled_vars)),
            axis=1
        )

        return tf.clip_by_value(combination_samples, clip_value_min=0.0, clip_value_max=1.0)

    @staticmethod
    @tf.function
    def Sequence(sampled_vars, states, dtype=tf.float64):
        """
        Compute the combination of sampled variables based on the given states.
        The states vector specifies the desired state for each variable:
        1 for occurrence, 0 for non-occurrence, and None to ignore the event.

        Args:
        - sampled_vars: A TensorFlow tensor of shape (num_samples, num_vars) containing sampled values.
        - states: A list where each element is 1 if the event occurred, 0 if the event did not occur,
                  and None if the event should be ignored.

        Returns:
        - A TensorFlow tensor containing the computed samples for the given combination.
        """
        states_tensor = tf.constant(states, dtype=dtype)
        return InverseCanopy.SequenceTF(sampled_vars, states_tensor)

    def _create_compute_y(self):
        y_sequences = self.targets['sequences']
        dtype = self.dtype
        input_signature = [
            tf.TensorSpec(shape=[self.num_samples, self.num_conditional_events], dtype=self.dtype,
                          name="sampled_conditional_probabilities"),
        ]

        @tf.function(input_signature=input_signature)
        def compute_y(sampled_vars):
            # Define a vectorized function to apply to each sequence
            def apply_sequence(seq):
                return InverseCanopy.SequenceTF(sampled_vars, seq)

            # Apply the vectorized function to each sequence
            y = tf.map_fn(apply_sequence, y_sequences, fn_output_signature=dtype)
            y = tf.clip_by_value(y, clip_value_min=0.0, clip_value_max=1.0)
            return y

        return compute_y

    def _create_compute_mu_sigma_from_sampled_distributions(self):

        input_signature = tf.TensorSpec(shape=[self.num_end_states, self.num_samples], dtype=self.dtype),

        @tf.function(input_signature=input_signature)
        def compute_mu_sigma_from_sampled_distributions(y_dists):
            log_y = tf.math.log(y_dists + 1e-30)
            mus = tf.math.reduce_mean(log_y, axis=1)
            sigmas = tf.math.reduce_std(log_y, axis=1)
            return log_y, mus, sigmas

        return compute_mu_sigma_from_sampled_distributions

    def _create_predict_end_state_likelihoods(self):

        input_signature = [
            tf.TensorSpec(shape=(self.num_conditional_events, ), dtype=self.dtype),
            tf.TensorSpec(shape=(self.num_conditional_events, ), dtype=self.dtype),
        ]

        @tf.function(input_signature=input_signature)
        def predict_end_state_likelihoods(mus, sigmas):
            predicted_samples = self.sample_from_distribution(mus, sigmas)  # num_samples
            y_pred_pdf = self.compute_y(predicted_samples)  # y_sequences
            return y_pred_pdf

        return predict_end_state_likelihoods

    def _create_mae_loss(self):

        input_signature = [
            tf.TensorSpec(shape=(self.num_end_states, self.num_samples), dtype=self.dtype),
            tf.TensorSpec(shape=(self.num_end_states, self.num_samples), dtype=self.dtype),
        ]

        @tf.function(input_signature=input_signature)
        def mae_loss(y_true, y_pred):
            # tf.print('mae_loss', y_true.shape, y_pred.shape)
            loss = tf.reduce_mean(tf.abs(y_true - y_pred), axis=1)
            return loss

        return mae_loss

    def _create_normalized_relative_logarithmic_error(self):

        log_y_true = self.targets['log_pdf']
        mu_y_true = self.targets['mus']
        sigma_y_true = self.targets['sigmas']

        input_signature = [
            tf.TensorSpec(shape=[self.num_end_states, self.num_samples], dtype=self.dtype),
        ]

        @tf.function(input_signature=input_signature)
        def normalized_relative_logarithmic_error(y_pred):
            log_y_pred, mu_y_pred, sigma_y_pred = self.compute_mu_sigma_from_sampled_distributions(y_pred)
            pdf_loss = tf.reduce_mean(self.mae_loss(log_y_true, log_y_pred))
            mu_loss = tf.reduce_mean(tf.abs(mu_y_true - mu_y_pred), axis=0)
            sigma_loss = tf.reduce_mean(tf.abs(sigma_y_true - sigma_y_pred), axis=0)

            # tf.print('pdf_loss', pdf_loss.shape, pdf_loss)
            # tf.print('mu_loss', mu_loss.shape, mu_loss)
            # tf.print('sigma_loss', sigma_loss.shape, sigma_loss)
            return pdf_loss + mu_loss + sigma_loss

        return normalized_relative_logarithmic_error

    def _create_normalized_relative_logarithmic_loss(self):

        input_signature = [
            tf.TensorSpec(shape=(self.num_conditional_events, ), dtype=self.dtype),
            tf.TensorSpec(shape=(self.num_conditional_events, ), dtype=self.dtype),
        ]

        @tf.function(input_signature=input_signature)
        def normalized_relative_logarithmic_loss(mus, sigmas):
            # tf.print(mus, sigmas, y_sequences)
            y_pred_pdf = self.predict_end_state_likelihoods(mus, sigmas)
            nrle = self.normalized_relative_logarithmic_error(y_pred_pdf)
            return nrle

        return normalized_relative_logarithmic_loss

    def optimization_step(self, optimizer):
        with tf.GradientTape() as tape:
            mus, sigmas = self.params.spread()
            loss = self.normalized_relative_logarithmic_loss(mus, sigmas)

        # Compute gradients with respect to model parameters that are variables
        gradients = tape.gradient(loss, [self.params['mus'], self.params['sigmas']])

        # Prepare (gradient, variable) pairs, applying the mask to each gradient
        grads_and_vars = []

        # mu masks
        masked_grad_mus = gradients[0] * tf.cast(self.trainable['mus'], dtype=gradients[0].dtype)
        grads_and_vars.append((masked_grad_mus, self.params['mus']))

        # sigma masks
        masked_grad_sigmas = gradients[1] * tf.cast(self.trainable['sigmas'], dtype=gradients[1].dtype)
        grads_and_vars.append((masked_grad_sigmas, self.params['sigmas']))

        # Apply the masked gradients to the optimizer
        optimizer.apply_gradients(grads_and_vars)

        # Convert mu, sigma to mean, std
        means_stds = self.params.to_meanstd()

        # Apply constraints in the original space
        clipped_means_stds = ModelInputs()
        clipped_means_stds['means'] = tf.clip_by_value(means_stds['means'], self.constraints['means'][0],
                                                       self.constraints['means'][1])
        clipped_means_stds['stds'] = tf.clip_by_value(means_stds['stds'], self.constraints['stds'][0],
                                                      self.constraints['stds'][1])

        # Convert back to mu, sigma
        constrained_mus_sigmas = clipped_means_stds.to_musigma()

        # Assign new mu, sigma
        self.params['mus'].assign(constrained_mus_sigmas['mus'])
        self.params['sigmas'].assign(constrained_mus_sigmas['sigmas'])

        #tf.print(self.params['mus'], self.params['sigmas'])
        self.params['mus'].assign(tf.minimum(self.params['mus'], -self.epsilon))
        self.params['mus'].assign(tf.maximum(self.params['mus'], tf.cast(-50, dtype=self.dtype)))
        self.params['sigmas'].assign(tf.minimum(self.params['sigmas'], tf.cast(6, dtype=self.dtype)))
        self.params['sigmas'].assign(tf.maximum(self.params['sigmas'], self.epsilon))

        # tf.debugging.check_numerics(self.params['mus'], "mus has NaNs")
        # tf.debugging.check_numerics(self.params['sigmas'], "sigmas has NaNs")

        self.params['mus'].assign(tf.where(tf.math.is_nan(self.params['mus']), tf.fill(self.params['sigmas'].shape, self.epsilon), self.params['mus']))
        self.params['sigmas'].assign(tf.where(tf.math.is_nan(self.params['sigmas']), tf.fill(self.params['sigmas'].shape, self.epsilon), self.params['sigmas']))
        return loss

    """
    end-user helper function
    """

    def fit(self, learning_rate=0.1, convergence_threshold=0.48, steps=1000):
        tf.print(f"learning_rate: {learning_rate}\n"
                 f"convergence_threshold: {convergence_threshold}\n"
                 f"max_steps: {steps}")

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate, amsgrad=False, epsilon=self.epsilon)
        return self.train_model(optimizer=optimizer, convergence_threshold=convergence_threshold, steps=steps)

    def train_model(self, optimizer, steps=1000, convergence_threshold=1e-10):
        for step in range(steps):
            loss = self.optimization_step(optimizer)

            if step % 100 == 0:
                # Print the losses along with the step number
                print(f"Step {step}: Loss = {loss.numpy():.16f}\n"
                      f"Mean-Std\n{self.params.to_meanstd()}")

            if loss.numpy() <= convergence_threshold:
                print(f"Stopping training at step {step} as loss reached the threshold of {convergence_threshold:.4e}")
                print(f"Step {step}: Loss = {loss.numpy():.16f}\n"
                      f"Mean-Std\n{self.params.to_meanstd()}")
                break
