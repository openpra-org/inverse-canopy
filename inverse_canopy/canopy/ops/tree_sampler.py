import tensorflow as tf

from inverse_canopy.canopy.ops.sampler import Sampler


class LogicTreeBroadcastSampler(Sampler):
    def __init__(self, logic_fn, num_inputs, num_outputs, num_batches, batch_size, sample_size,
                 bitpack_dtype: tf.uint8,
                 sampler_dtype: tf.float32,
                 acc_dtype: tf.float32, name=None):
        tf.config.run_functions_eagerly(False)
        super().__init__(name=name)

        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._num_batches = num_batches
        self._batch_size = batch_size
        self._sample_size = sample_size
        self._bitpack_dtype = bitpack_dtype
        self._sampler_dtype = sampler_dtype
        self._acc_dtype = acc_dtype

        self._num_sampled_bits_in_batch = tf.constant(value=tf.cast(self._batch_size * self._sample_size * super(LogicTreeBroadcastSampler, self)._compute_bits_in_dtype(self._bitpack_dtype), dtype=acc_dtype), dtype=acc_dtype)

        self._mse_loss = tf.function(
            func=lambda y_true, y_pred: super(LogicTreeBroadcastSampler, self)._mse_loss(y_true, y_pred, dtype=self._acc_dtype),
            input_signature=[
                tf.TensorSpec(shape=[self._batch_size, self._sample_size], dtype=self._acc_dtype, name='y_true'),
                tf.TensorSpec(shape=[self._batch_size, self._sample_size], dtype=self._acc_dtype, name='y_pred'),
            ],
            jit_compile=True
        )

        self._generate_bernoulli_batch = tf.function(
            func=lambda probs, seed: super(LogicTreeBroadcastSampler, self)._generate_bernoulli(
                probs=probs,
                n_sample_packs_per_probability=tf.constant(value=self._sample_size, dtype=tf.int32) ,
                bitpack_dtype=self._bitpack_dtype,
                dtype=self._sampler_dtype,
                seed=seed
            ),
            input_signature=[
                tf.TensorSpec(shape=[self._num_inputs, self._batch_size], dtype=self._sampler_dtype, name='probs'),
                tf.TensorSpec(shape=[], dtype=tf.int32, name='seed'),
            ],
            jit_compile=True
        )

        self._generate_bernoulli_broadcast_no_batch = tf.function(
            func=lambda probs, seed: super(LogicTreeBroadcastSampler, self)._generate_bernoulli(
                probs=tf.broadcast_to(tf.expand_dims(probs, axis=1), [self._num_inputs, self._batch_size]),
                n_sample_packs_per_probability=tf.constant(value=self._sample_size, dtype=tf.int32) ,
                bitpack_dtype=self._bitpack_dtype,
                dtype=self._sampler_dtype,
                seed=seed
            ),
            input_signature=[
                tf.TensorSpec(shape=[self._num_inputs], dtype=self._sampler_dtype, name='probs'),
                tf.TensorSpec(shape=[], dtype=tf.int32, name='seed'),
            ],
            jit_compile=True
        )

        self._logic_fn = tf.function(
            func=logic_fn,
            input_signature=[
                tf.TensorSpec(shape=[self._num_inputs, self._batch_size, self._sample_size], dtype=self._bitpack_dtype, name='sampled_inputs'),
            ],
            jit_compile=True
        )

        self._count = tf.function(
            func=lambda x: super(LogicTreeBroadcastSampler, self)._count_one_bits(
                x=x,
                axis=None,
                dtype=tf.uint32,
            ),
            input_signature=[
                tf.TensorSpec(shape=[self._batch_size, self._sample_size], dtype=self._bitpack_dtype, name='raw_bits'),
            ],
            jit_compile=True
        )

        self._tally = tf.function(
            func=lambda means: super(LogicTreeBroadcastSampler, self)._p95_ci(
                means=means,
                total=self._num_sampled_bits_in_batch,
                dtype=self._acc_dtype,
            ),
            input_signature=[
                tf.TensorSpec(shape=[], dtype=self._acc_dtype, name='means'),
            ],
            jit_compile=True
        )

    def generate(self, probs, seed=372):
        return self._generate_bernoulli_broadcast_no_batch(probs=probs, seed=seed,)

    def eval(self, probs, seed=372):
        input_packed_bits_ = self._generate_bernoulli_broadcast_no_batch(probs=probs, seed=seed,)
        output_packed_bits_ = self._logic_fn(input_packed_bits_)
        return output_packed_bits_

    def eval_fn(self, fn, probs, seed=372):
        input_packed_bits_ = self._generate_bernoulli_broadcast_no_batch(probs=probs, seed=seed,)
        output_packed_bits_ = fn(input_packed_bits_)
        return output_packed_bits_

    def count(self, probs, seed=372):
        input_packed_bits_ = self._generate_bernoulli_broadcast_no_batch(probs=probs, seed=seed,)
        output_packed_bits_ = self._logic_fn(input_packed_bits_)
        ones_ = self._count(output_packed_bits_)
        return ones_

    def expectation(self, probs, seed=372):
        input_packed_bits_ = self._generate_bernoulli_broadcast_no_batch(probs=probs, seed=seed,)
        output_packed_bits_ = self._logic_fn(input_packed_bits_)
        ones_ = self._count(output_packed_bits_)
        means_ = tf.cast(ones_, dtype=self._acc_dtype) / self._num_sampled_bits_in_batch
        return means_

    def tally(self, probs, seed=372):
        input_packed_bits_ = self._generate_bernoulli_broadcast_no_batch(probs=probs, seed=seed,)
        output_packed_bits_ = self._logic_fn(input_packed_bits_)
        ones_ = self._count(output_packed_bits_)
        means_ = tf.cast(ones_, dtype=self._acc_dtype) / self._num_sampled_bits_in_batch
        p05_, p95_ = self._tally(means_)
        return p05_, means_, p95_

    def tally_from_samples(self, samples):
        ones_ = self._count(samples)
        means_ = tf.cast(ones_, dtype=self._acc_dtype) / self._num_sampled_bits_in_batch
        p05_, p95_ = self._tally(means_)
        return p05_, means_, p95_