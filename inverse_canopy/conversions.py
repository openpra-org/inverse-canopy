import tensorflow as tf


def base_10(base_e_power, dtype=tf.float64):
    """
    Convert base 10 values to base e
    """
    base_e_power = tf.cast(base_e_power, dtype)
    ten = tf.cast(10.0, dtype)
    return base_e_power * tf.math.log(ten)


def base_e(base_10_power, dtype=tf.float64):
    """
    Convert base e values to base 10
    """
    base_10_power = tf.cast(base_10_power, dtype)
    ten = tf.cast(10.0, dtype)
    return -base_10_power / tf.math.log(ten)


def compute_mu_sigma(mean, std, dtype=tf.float64):
    """
    Compute the mu and sigma parameters of a lognormal distribution from its mean and standard deviation.

    Args:
    - mean: A tensor representing the mean of the lognormal distribution.
    - std: A tensor representing the standard deviation of the lognormal distribution.

    Returns:
    - A tuple of tensors representing the mu and sigma parameters of the lognormal distribution.
    """
    two = tf.cast(2.0, dtype)
    one = tf.cast(1.0, dtype)
    mean = tf.cast(mean, dtype)
    std = tf.cast(std, dtype)
    mu = tf.math.log(mean**two / tf.sqrt(std**two + mean**two))
    sigma = tf.sqrt(tf.math.log(one + std**two / mean**two))
    return mu, sigma


def compute_mean_std(mu, sigma, dtype=tf.float64):
    """
    Compute the mean and standard deviation of a lognormal distribution from its mu and sigma parameters.

    Args:
    - mu: A tensor representing the mu parameter of the lognormal distribution.
    - sigma: A tensor representing the sigma parameter of the lognormal distribution.

    Returns:
    - A tuple of tensors representing the mean and standard deviation of the lognormal distribution.
    """
    mu = tf.cast(mu, dtype)
    sigma = tf.cast(sigma, dtype)
    two = tf.cast(2.0, dtype)
    one = tf.cast(1.0, dtype)
    mean = tf.exp(mu + sigma**two / two)
    std = mean * tf.sqrt(tf.exp(sigma ** 2) - one)
    return mean, std


def compute_mean_error_factor_from_mu_sigma(mu, sigma, dtype=tf.float64):
    mu = tf.cast(mu, dtype)
    sigma = tf.cast(sigma, dtype)
    two = tf.cast(2.0, dtype)
    z = tf.cast(1.6448536269514722, dtype)
    mean = tf.exp(mu + sigma**two / two)
    error_factor = tf.exp(z * sigma)
    return mean, error_factor


def compute_5th_mean_95th_from_mu_sigma(mu, sigma, dtype=tf.float64):
    mean, error_factor = compute_mean_error_factor_from_mu_sigma(mu, sigma, dtype)
    median = tf.exp(mu)
    p95 = median * error_factor
    p05 = median / error_factor
    return p05, mean, p95


def compute_mu_sigma_from_5th_mean_95th(p05, mean, p95, dtype=tf.float64):
    z = tf.cast(3.2897071310939356, dtype)
    sigma = tf.math.log(p95/p05) / z
    two = tf.cast(2.0, dtype)
    mu = tf.math.log(mean) - (sigma**two / two)
    return mu, sigma


def constrain_mu(unconstrained_mu, lower_bound=-30.0, upper_bound=0.0, dtype=tf.float64):
    """
    Map an unconstrained mu parameter to a constrained space defined by the given bounds using a sigmoid function.

    Args:
    - unconstrained_mu: A tensor representing the unconstrained mu parameter.
    - lower_bound: A float representing the lower bound of the constrained space for mu.
    - upper_bound: A float representing the upper bound of the constrained space for mu.

    Returns:
    - A tensor representing the constrained mu parameter.
    """
    lower_bound = tf.cast(lower_bound, dtype)
    upper_bound = tf.cast(upper_bound, dtype)
    unconstrained_mu = tf.cast(unconstrained_mu, dtype)
    constrained_mu = lower_bound + (upper_bound - lower_bound) * tf.sigmoid(unconstrained_mu)
    return constrained_mu


def unconstrain_mu(constrained_mu, lower_bound=-30, upper_bound=0, dtype=tf.float64):
    """
    Map a constrained mu parameter back to the unconstrained space using the inverse of the sigmoid function.

    Args:
    - constrained_mu: A tensor representing the constrained mu parameter.
    - lower_bound: A float representing the lower bound of the constrained space for mu.
    - upper_bound: A float representing the upper bound of the constrained space for mu.

    Returns:
    - A tensor representing the unconstrained mu parameter.
    """
    lower_bound = tf.cast(lower_bound, dtype)
    upper_bound = tf.cast(upper_bound, dtype)
    constrained_mu = tf.cast(constrained_mu, dtype)
    unconstrained_mu = tf.math.log((constrained_mu - lower_bound) / (upper_bound - constrained_mu))
    return unconstrained_mu


def constrain_sigma(unconstrained_sigma, lower_bound=0.0, upper_bound=20.0, epsilon=tf.keras.backend.epsilon(), dtype=tf.float64):
    """
    Map an unconstrained sigma parameter to a constrained space defined by the given bounds using a softplus function.

    Args:
    - unconstrained_sigma: A tensor representing the unconstrained sigma parameter.
    - lower_bound: A float representing the lower bound of the constrained space for sigma.
    - upper_bound: A float representing the upper bound of the constrained space for sigma.
    - epsilon: avoid divide by zero without magnifiying the output too much.

    Returns:
    - A tensor representing the constrained sigma parameter.
    """
    # Ensure that the unconstrained_sigma is positive using softplus and then scale to the (lower_bound, upper_bound)
    lower_bound = tf.cast(lower_bound, dtype)
    upper_bound = tf.cast(upper_bound, dtype)
    unconstrained_sigma = tf.cast(unconstrained_sigma, dtype)
    epsilon = tf.cast(epsilon, dtype)
    scale = tf.nn.softplus(upper_bound) - tf.nn.softplus(lower_bound) + epsilon
    constrained_sigma = lower_bound + (upper_bound - lower_bound) * (tf.nn.softplus(unconstrained_sigma) - tf.nn.softplus(lower_bound)) / scale
    return constrained_sigma


def unconstrain_sigma(constrained_sigma, lower_bound=0.0, upper_bound=20.0, epsilon=tf.keras.backend.epsilon(), dtype=tf.float64):
    """
    Map a constrained sigma parameter back to the unconstrained space using the inverse of the softplus function.

    Args:
    - constrained_sigma: A tensor representing the constrained sigma parameter.
    - lower_bound: A float representing the lower bound of the constrained space for sigma.
    - upper_bound: A float representing the upper bound of the constrained space for sigma.
    - epsilon: avoid divide by zero without magnifiying the output too much.

    Returns:
    - A tensor representing the unconstrained sigma parameter.
    """
    lower_bound = tf.cast(lower_bound, dtype)
    upper_bound = tf.cast(upper_bound, dtype)
    constrained_sigma = tf.cast(constrained_sigma, dtype)
    epsilon = tf.cast(epsilon, dtype)
    scale = tf.nn.softplus(upper_bound) - tf.nn.softplus(lower_bound) + epsilon
    one = tf.cast(1.0, dtype)
    unconstrained_sigma = tf.math.log(tf.exp((constrained_sigma - lower_bound) * scale / (upper_bound - lower_bound) + tf.nn.softplus(lower_bound)) - one)
    return unconstrained_sigma


def compute_rmsle(predicted_means, validation_means, dtype=tf.float64):
    """
    Compute the Root Mean Squared Logarithmic Error between predicted and validation means.

    Args:
    - predicted_means: A tf.Tensor containing the predicted means.
    - validation_means: A tf.Tensor containing the validation means.

    Returns:
    - The RMSLE as a tf.Tensor.
    """
    # Ensure the tensors are of the same type
    predicted_means = tf.cast(predicted_means, dtype)
    validation_means = tf.cast(validation_means, dtype)

    # Adding 1 to both tensors to avoid log(0)
    one = tf.cast(1.0, dtype)
    log_predicted = tf.math.log(predicted_means + one)
    log_validation = tf.math.log(validation_means + one)

    # Compute the squared logarithmic differences
    squared_log_diff = tf.square(log_predicted - log_validation)

    # Compute the mean of the squared logarithmic differences
    msle = tf.reduce_mean(squared_log_diff)

    # Compute the square root of MSLE to get RMSLE
    rmsle = tf.sqrt(msle)

    return rmsle
