"""
This module defines classes for handling model parameters, inputs, and outputs,
with functionality to convert between different representations.
"""

import tensorflow as tf
from .lognormal_utils import compute_mu_sigma, compute_mean_std


class ModelOutputs(dict):
    """
    A class to represent the outputs of a model, including means and standard deviations.
    """
    means: tf.Tensor
    stds: tf.Tensor

    def __init__(self, means=None, stds=None):
        """
        Initialize the ModelOutputs instance.

        Args:
            means (tf.Tensor, optional): The means of the model outputs. Defaults to None.
            stds (tf.Tensor, optional): The standard deviations of the model outputs. Defaults to None.
        """
        super().__init__(means=means, stds=stds)
        if stds is None:
            stds = []
        if means is None:
            means = []
        self['means'] = means
        self['stds'] = stds

    def to_musigma(self, dtype=tf.float64) -> 'ModelParams':
        """
        Convert means and standard deviations to mu and sigma parameters.

        Args:
            dtype (tf.DType, optional): The data type for the conversion. Defaults to tf.float64.

        Returns:
            ModelParams: An instance of ModelParams with mu and sigma values.
        """
        mus, sigmas = compute_mu_sigma(self['means'], self['stds'], dtype=dtype)
        return ModelParams(mus=mus, sigmas=sigmas)

    def __repr__(self):
        means_str = ", ".join([f"{m:.2e}" for m in self['means'].numpy()])
        stds_str = ", ".join([f"{s:.2e}" for s in self['stds'].numpy()])
        return f"ModelOutputs(means=[{means_str}], stds=[{stds_str}])"

    def __str__(self):
        ret_str = ""
        for i, (mean, std) in enumerate(zip(self['means'].numpy(), self['stds'].numpy())):
            ret_str += f"[{i}]: Mean={mean:.2e}, Std={std:.2e}\n"
        return ret_str

    def spread(self):
        """
        Get the means and standard deviations as separate entities.

        Returns:
            tuple: A tuple containing the means and standard deviations.
        """
        return self['means'], self['stds']


class ModelInputs(ModelOutputs):
    """
    A class to represent the inputs to a model, inheriting from ModelOutputs.
    """
    def __repr__(self):
        means_str = ", ".join([f"{m:.2e}" for m in self['means'].numpy()])
        stds_str = ", ".join([f"{s:.2e}" for s in self['stds'].numpy()])
        return f"ModelInputs(means=[{means_str}], stds=[{stds_str}])"


class ModelParams(dict):
    """
    A class to represent the parameters of a model, including mus and sigmas.
    """
    mus: tf.Variable
    sigmas: tf.Variable

    def __init__(self, mus=None, sigmas=None):
        """
        Initialize the ModelParams instance.

        Args:
            mus (tf.Variable, optional): The mu parameters of the model. Defaults to None.
            sigmas (tf.Variable, optional): The sigma parameters of the model. Defaults to None.
        """
        super().__init__()
        if sigmas is None:
            sigmas = []
        if mus is None:
            mus = []
        self['mus'] = mus
        self['sigmas'] = sigmas

    def validate(self):
        """
        Validate the model parameters. Currently, this method is a placeholder.
        """

    def spread(self):
        """
        Get the mus and sigmas as separate entities.

        Returns:
            tuple: A tuple containing the mus and sigmas.
        """
        return self['mus'], self['sigmas']

    def to_meanstd(self, dtype=tf.float64) -> ModelOutputs:
        """
        Convert mu and sigma parameters to means and standard deviations.

        Args:
            dtype (tf.DType, optional): The data type for the conversion. Defaults to tf.float64.

        Returns:
            ModelOutputs: An instance of ModelOutputs with means and standard deviations.
        """
        means, stds = compute_mean_std(self['mus'], self['sigmas'], dtype=dtype)
        return ModelOutputs(means=means, stds=stds)

    def __repr__(self):
        mus_str = ", ".join([f"{m:.2e}" for m in self['mus'].numpy()])
        sigmas_str = ", ".join([f"{s:.2e}" for s in self['sigmas'].numpy()])
        return f"ModelParams(mus=[{mus_str}], sigmas=[{sigmas_str}])"

    def __str__(self):
        ret_str = ""
        for i, (mu, sigma) in enumerate(zip(self['mus'].numpy(), self['sigmas'].numpy())):
            ret_str += f"[{i}]: mu={mu:.2e}, sigma={sigma:.2e}\n"
        return ret_str


class TrainableParams(ModelParams):
    """
    A class to represent trainable parameters of a model, inheriting from ModelParams.
    """
    mus: tf.constant
    sigmas: tf.constant

    def __init__(self, mus=None, sigmas=None):
        """
        Initialize the TrainableParams instance.

        Args:
            mus (tf.constant, optional): The trainable mu parameters of the model. Defaults to None.
            sigmas (tf.constant, optional): The trainable sigma parameters of the model. Defaults to None.
        """
        super().__init__(mus, sigmas)
        if sigmas is None:
            sigmas = []
        if mus is None:
            mus = []
        self['mus'] = mus
        self['sigmas'] = sigmas

    def __repr__(self):
        mus_str = ", ".join([f"{m:.2e}" for m in self['mus'].numpy()])
        sigmas_str = ", ".join([f"{s:.2e}" for s in self['sigmas'].numpy()])
        return f"TrainableParams(mus=[{mus_str}], sigmas=[{sigmas_str}])"
