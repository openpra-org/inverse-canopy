import unittest
import tensorflow as tf
from inverse_canopy.lognormal_utils import compute_mu_sigma, compute_mean_std, compute_5th_mean_95th_from_mu_sigma, compute_mu_sigma_from_5th_mean_95th, compute_mean_error_factor_from_mu_sigma


class TestLognormalUtils(unittest.TestCase):

    def test_mu_sigma_to_mean_std_and_back(self):
        mu = tf.constant(1.0, dtype=tf.float64)
        sigma = tf.constant(0.5, dtype=tf.float64)
        mean, std = compute_mean_std(mu, sigma)
        mu_back, sigma_back = compute_mu_sigma(mean, std)

        self.assertAlmostEqual(mu.numpy(), mu_back.numpy(), places=5)
        self.assertAlmostEqual(sigma.numpy(), sigma_back.numpy(), places=5)

    def test_mean_std_to_mu_sigma_and_back(self):
        mean = tf.constant(10.0, dtype=tf.float64)
        std = tf.constant(3.0, dtype=tf.float64)
        mu, sigma = compute_mu_sigma(mean, std)
        mean_back, std_back = compute_mean_std(mu, sigma)

        self.assertAlmostEqual(mean.numpy(), mean_back.numpy(), places=5)
        self.assertAlmostEqual(std.numpy(), std_back.numpy(), places=5)

    def test_mu_sigma_to_percentiles_and_back(self):
        mu = tf.constant(0.0, dtype=tf.float64)
        sigma = tf.constant(0.25, dtype=tf.float64)
        p05, mean, p95 = compute_5th_mean_95th_from_mu_sigma(mu, sigma)
        mu_back, sigma_back = compute_mu_sigma_from_5th_mean_95th(p05, mean, p95)

        self.assertAlmostEqual(mu.numpy(), mu_back.numpy(), places=5)
        self.assertAlmostEqual(sigma.numpy(), sigma_back.numpy(), places=5)

    def test_mu_sigma_to_mean_error_factor_and_validate(self):
        mu = tf.constant(0.0, dtype=tf.float64)
        sigma = tf.constant(0.25, dtype=tf.float64)
        mean, error_factor = compute_mean_error_factor_from_mu_sigma(mu, sigma)

        two = tf.cast(2.0, dtype=tf.float64)
        z = tf.cast(1.6448536269514722, dtype=tf.float64)  # Approximation for 95% confidence interval of normal distribution

        # Calculate expected values
        expected_mean = tf.exp(mu + sigma ** two / two)
        expected_error_factor = tf.exp(z * sigma)

        self.assertAlmostEqual(mean.numpy(), expected_mean.numpy(), places=5)
        self.assertAlmostEqual(error_factor.numpy(), expected_error_factor.numpy(), places=5)

    def test_error_factor_consistency(self):
        mu = tf.constant(0.0, dtype=tf.float64)
        sigma = tf.constant(0.25, dtype=tf.float64)
        mean, error_factor = compute_mean_error_factor_from_mu_sigma(mu, sigma)

        # Derive std from mean and error factor
        one = tf.cast(1.0, dtype=tf.float64)
        z = tf.cast(1.6448536269514722, dtype=tf.float64)  # Approximation for 95% confidence interval of normal distribution
        std = mean * (tf.exp(z * sigma) - one) / z

        # Compute mu and sigma back from mean and std
        mu_back, sigma_back = compute_mu_sigma(mean, std)

        self.assertAlmostEqual(mu.numpy(), mu_back.numpy(), places=5)
        self.assertAlmostEqual(sigma.numpy(), sigma_back.numpy(), places=5)


if __name__ == '__main__':
    unittest.main()