import tensorflow as tf
from .lognormal_utils import compute_mu_sigma, compute_mean_std


class ModelOutputs(dict):
    means: tf.Tensor
    stds: tf.Tensor

    def __init__(self, means=None, stds=None):
        super().__init__(means=means, stds=stds)
        if stds is None:
            stds = []
        if means is None:
            means = []
        self['means'] = means
        self['stds'] = stds

    def to_musigma(self, dtype=tf.float64) -> 'ModelParams':
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
        return self['means'], self['stds']


class ModelInputs(ModelOutputs):
    def __repr__(self):
        means_str = ", ".join([f"{m:.2e}" for m in self['means'].numpy()])
        stds_str = ", ".join([f"{s:.2e}" for s in self['stds'].numpy()])
        return f"ModelInputs(means=[{means_str}], stds=[{stds_str}])"


class ModelParams(dict):
    mus: tf.Variable
    sigmas: tf.Variable

    def __init__(self, mus=None, sigmas=None):
        super().__init__()
        if sigmas is None:
            sigmas = []
        if mus is None:
            mus = []
        self['mus'] = mus
        self['sigmas'] = sigmas

    def validate(self):
        pass

    def spread(self):
        return self['mus'], self['sigmas']

    def to_meanstd(self, dtype=tf.float64) -> ModelOutputs:
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
    mus: tf.constant
    sigmas: tf.constant

    def __init__(self, mus=None, sigmas=None):
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
