"""
Module: lognormal_utils

This module contains functions for computing the mu and sigma parameters of a lognormal distribution from its mean and
standard deviation, as well as functions for converting between different parameter representations.

Functions:
- compute_mu_sigma: Compute the mu and sigma parameters of a lognormal distribution from its mean and standard deviation
- compute_mean_std: Compute the mean and standard deviation of a lognormal distribution from its mu and sigma parameters
- compute_mean_error_factor_from_mu_sigma: Compute the mean & error factor of a lognormal distribution from mu & sigma.
- compute_5th_mean_95th_from_mu_sigma: Compute the 5th, mean, and 95th percentile of a lognormal distribution from
its mu & sigma.
- compute_mu_sigma_from_5th_mean_95th: Compute the mu & sigma parameters of a lognormal distribution from its 5th,
 mean, and 95th percentile.

Dependencies:
- tensorflow
"""
import tensorflow as tf


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
    """
    Compute the mean and error factor of a lognormal distribution from its mu and sigma parameters.

    Args:
    - mu: A tensor representing the mu parameter of the lognormal distribution.
    - sigma: A tensor representing the sigma parameter of the lognormal distribution.

    Returns:
    - A tuple of tensors representing the mean and error factor of the lognormal distribution.
    """
    mu = tf.cast(mu, dtype)
    sigma = tf.cast(sigma, dtype)
    two = tf.cast(2.0, dtype)
    z = tf.cast(1.6448536269514722, dtype)
    mean = tf.exp(mu + sigma**two / two)
    error_factor = tf.exp(z * sigma)
    return mean, error_factor


def compute_5th_mean_95th_from_mu_sigma(mu, sigma, dtype=tf.float64):
    """
    Compute the 5th percentile, mean, and 95th percentile of a lognormal distribution from its mu and sigma parameters.

    Args:
    - mu: A tensor representing the mu parameter of the lognormal distribution.
    - sigma: A tensor representing the sigma parameter of the lognormal distribution.

    Returns:
    - A tuple of tensors representing the 5th percentile, mean, and 95th percentile of the lognormal distribution.
    """
    mean, error_factor = compute_mean_error_factor_from_mu_sigma(mu, sigma, dtype)
    median = tf.exp(mu)
    p95 = median * error_factor
    p05 = median / error_factor
    return p05, mean, p95


def compute_mu_sigma_from_5th_mean_95th(p05, mean, p95, dtype=tf.float64):
    """
    Compute the mu and sigma parameters of a lognormal distribution from its 5th percentile, mean, and 95th percentile.

    Args:
    - p05: A tensor representing the 5th percentile of the lognormal distribution.
    - mean: A tensor representing the mean of the lognormal distribution.
    - p95: A tensor representing the 95th percentile of the lognormal distribution.

    Returns:
    - A tuple of tensors representing the mu and sigma parameters of the lognormal distribution.
    """
    z = tf.cast(3.2897071310939356, dtype)
    sigma = tf.math.log(p95/p05) / z
    two = tf.cast(2.0, dtype)
    mu = tf.math.log(mean) - (sigma**two / two)
    return mu, sigma
