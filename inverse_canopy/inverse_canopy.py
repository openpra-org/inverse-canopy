import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.constraints import NonNeg
from .params import ModelInputs, TrainableParams, ModelParams, ModelOutputs


class ProbabilityModel(tf.Module):
    def __init__(self, initial_guess: ModelInputs, trainable_mask: TrainableParams, constraints: ModelInputs,
                 dtype=tf.float64):
        super().__init__()

        # initialize internal model params
        self.params: ModelParams = initial_guess.to_musigma()
        self.params['mus'] = tf.Variable(self.params['mus'], dtype=dtype)
        self.params['sigmas'] = tf.Variable(self.params['sigmas'], dtype=dtype, constraint=NonNeg())

        # set the trainable boolean masks
        self.trainable: TrainableParams = trainable_mask
        self.trainable['mus'] = tf.constant(self.trainable['mus'], dtype)
        self.trainable['sigmas'] = tf.constant(self.trainable['sigmas'], dtype)

        # clipping constraints on mean, std
        self.constraints = constraints
        self.constraints['means'] = tf.constant(self.constraints['means'], dtype)
        self.constraints['stds'] = tf.constant(self.constraints['stds'], dtype)

        tf.print('params', self.params['mus'], self.params['sigmas'])
        tf.print('mask', self.trainable['mus'], self.trainable['sigmas'])
        tf.print('constraints', self.constraints['means'], self.constraints['stds'])

    @staticmethod
    @tf.function
    def sample_from_distribution(mus, sigmas, num_samples, low=0.0, high=1.0, epsilon=1e-30, dtype=tf.float64):
        low = tf.cast(low, dtype)
        high = tf.cast(high, dtype)
        low_normal = tf.math.log(low + epsilon)  # A small number close to 0
        high_normal = tf.math.log(high)  # Log(1) = 0

        # @tf.function
        def sample_single(mu_sigma):
            # tf.print("loc:", mu_sigma[0], "scale:", mu_sigma[1], "low:", low_normal, "high:", high_normal)
            truncated = tfp.distributions.TruncatedNormal(loc=mu_sigma[0], scale=(mu_sigma[1] + epsilon),
                                                          low=low_normal, high=high_normal)
            return tf.exp(truncated.sample(num_samples))

        # Stack 'mus' and 'sigmas' along a new axis to create pairs
        mus_sigmas = tf.stack([mus, sigmas], axis=1)

        # Use 'tf.map_fn' to apply 'sample_single' across all 'mu_sigma' pairs
        samples = tf.map_fn(sample_single, mus_sigmas, fn_output_signature=dtype)

        # Transpose the samples to adjust the shape if necessary
        return tf.transpose(samples)

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
        return ProbabilityModel.SequenceTF(sampled_vars, states_tensor)

    @staticmethod
    @tf.function
    def compute_Y(sampled_vars, sequences, dtype=tf.float64):
        # Convert sequences list to a tensor if it's not already one
        sequences_tensor = tf.convert_to_tensor(sequences, dtype=dtype)

        # Define a vectorized function to apply to each sequence
        def apply_sequence(seq):
            return ProbabilityModel.SequenceTF(sampled_vars, seq)

        # Apply the vectorized function to each sequence
        y = tf.map_fn(apply_sequence, sequences_tensor, fn_output_signature=dtype)
        y = tf.clip_by_value(y, clip_value_min=0.0, clip_value_max=1.0)
        return y

    @staticmethod
    @tf.function
    def compute_mu_sigma_from_sampled_distributions(y_dists, axis=1, epsilon=1e-30):
        log_y = tf.math.log(y_dists + epsilon)
        mus = tf.math.reduce_mean(log_y, axis=axis)
        sigmas = tf.math.reduce_std(log_y, axis=axis)
        return log_y, mus, sigmas

    @staticmethod
    @tf.function
    def predict_end_state_likelihoods(mus, sigmas, y_sequences, num_samples=100000) -> ModelOutputs:
        predicted_samples = ProbabilityModel.sample_from_distribution(mus, sigmas, num_samples)
        y_pred_pdf = ProbabilityModel.compute_Y(predicted_samples, y_sequences)
        return y_pred_pdf

    @staticmethod
    @tf.function
    def mae_loss(y_true, y_pred, axis=1):
        loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        return loss

    @staticmethod
    @tf.function
    def normalized_relative_logarithmic_error(y_true, y_pred):
        # Compute the logarithm of the distribution
        log_y_true, mu_y_true, sigma_y_true = ProbabilityModel.compute_mu_sigma_from_sampled_distributions(y_true)
        log_y_pred, mu_y_pred, sigma_y_pred = ProbabilityModel.compute_mu_sigma_from_sampled_distributions(y_pred)

        pdf_loss = ProbabilityModel.mae_loss(log_y_true, log_y_pred)
        mu_loss = ProbabilityModel.mae_loss(mu_y_true, mu_y_pred)
        sigma_loss = ProbabilityModel.mae_loss(sigma_y_true, sigma_y_pred)

        combined_loss = pdf_loss + mu_loss + sigma_loss
        return combined_loss

    @staticmethod
    @tf.function
    def rmsle(y_true, y_pred, epsilon=1e-30):
        """
        Root Mean Squared Logarithmic Error (RMSLE) loss function.

        Args:
        - y_true: TensorFlow tensor containing the true values.
        - y_pred: TensorFlow tensor containing the predicted values.
        - epsilon_value: A small value to ensure numerical stability.

        Returns:
        - A tensor containing the computed RMSLE.
        """
        # Compute the logarithm of y_true and y_pred, adding epsilon to avoid log(0)
        log_y_true = tf.math.log(y_true + epsilon)
        log_y_pred = tf.math.log(y_pred + epsilon)

        # Compute the squared difference between the logarithms
        squared_log_error = tf.square(log_y_true - log_y_pred)

        # Compute the mean of the squared log errors
        mean_squared_log_error = tf.reduce_mean(squared_log_error)

        # Return the square root of the mean squared log error
        return tf.sqrt(mean_squared_log_error)

    @staticmethod
    @tf.function
    def variance_loss(y_true_std, y_pred_std):
        return tf.reduce_mean(tf.square(y_true_std - y_pred_std))

    @tf.function
    def normalized_relative_logarithmic_loss(self, y_observed_pdf, y_sequences, num_samples):
        mus, sigmas = self.params.spread()
        y_pred_pdf = self.predict_end_state_likelihoods(mus, sigmas, y_sequences, num_samples)
        nrle = ProbabilityModel.normalized_relative_logarithmic_error(y_pred_pdf, y_observed_pdf)
        return nrle, y_pred_pdf

    @tf.function
    def optimization_step(self, y_observed_pdf, y_sequences, optimizer, num_samples, epsilon=1e-30, dtype=tf.float64):
        with tf.GradientTape() as tape:
            loss, y_pred_pdf = self.normalized_relative_logarithmic_loss(y_observed_pdf, y_sequences, num_samples)

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

        tf.print(self.params['mus'], self.params['sigmas'])
        self.params['mus'].assign(tf.minimum(self.params['mus'], -epsilon))
        self.params['mus'].assign(tf.maximum(self.params['mus'], tf.cast(-50, dtype=dtype)))
        self.params['sigmas'].assign(tf.minimum(self.params['sigmas'], tf.cast(6, dtype=dtype)))
        self.params['sigmas'].assign(tf.maximum(self.params['sigmas'], epsilon))

        return loss, y_pred_pdf, gradients

    def train_model(self, y_observed_pdf, y_sequences, optimizer, steps=100,
                    num_samples=10000, convergence_threshold=1e-10, max_steps=None):

        for step in range(steps):
            loss, y_pred_pdf, gradients = self.optimization_step(y_observed_pdf, y_sequences, optimizer, num_samples)

            if step % 100 == 0:
                # Print the losses along with the step number
                print(f"Step {step}: Loss = {loss.numpy():.16f}\n"
                      f"Mean-Std\n{self.params.to_meanstd()}")

            if loss.numpy() <= convergence_threshold or (max_steps is not None and step >= max_steps):
                print(f"Stopping training at step {step} as loss reached the threshold of {convergence_threshold:.4e}")
                print(f"Step {step}: Loss = {loss.numpy():.16f}\n"
                      f"Mean-Std\n{self.params.to_meanstd()}")
                break
