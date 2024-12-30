import os
import sys
from typing import Dict, List
import tensorflow as tf

from inverse_canopy.canopy.dag import topological_sort, presort_event_nodes
from inverse_canopy.canopy.event import Node, Gate, BasicEvent
from inverse_canopy.canopy.ops.sampler import Sampler
from inverse_canopy.canopy.parser import parse_fault_tree


def generate_basic_event_samples(rng:tf.random.Generator, probabilities: List[float], batch_size: int, sample_size: int, sampler_dtype=tf.float32, bitpack_dtype=tf.uint8) -> tf.Tensor:
    # Create a tensor of probabilities with shape [num_events, batch_size, sample_size]
    probs = tf.constant(probabilities, dtype=sampler_dtype)
    num_events = tf.shape(probs)[0]
    probs_broadcast = tf.broadcast_to(tf.expand_dims(probs, axis=1), [num_events, batch_size])
    samples = Sampler.generate_bernoulli(rng=rng,
                                          probs=probs_broadcast,
                                          n_sample_packs_per_probability=sample_size,
                                          bitpack_dtype=bitpack_dtype,
                                          dtype=sampler_dtype)
    return samples  # Shape: [num_events, batch_size, sample_size]


@tf.function(jit_compile=True)
def build_tf_graph(presorted_nodes, rng: tf.random.Generator, batch_size: tf.int32, sample_size: tf.int32):

    basic_event_probs: List[float] = presorted_nodes["basic_events"]["probabilities"]

    # Generate samples for all basic events at once
    basic_event_samples: tf.Tensor = generate_basic_event_samples(rng=rng,
                                                       probabilities=basic_event_probs,
                                                       batch_size=batch_size,
                                                       sample_size=sample_size)

    sorted_basic_events: List[BasicEvent] = presorted_nodes["basic_events"]["sorted"]
    basic_event_probability_indices: Dict[str, int] = presorted_nodes["basic_events"]["probability_indices"]

    # Map basic event outputs
    for basic_event in sorted_basic_events:
        index = basic_event_probability_indices[basic_event.name]
        basic_event.output = basic_event_samples[index]

    # apply the tensorflow bitwise op
    sorted_gates: List[Gate] = presorted_nodes["gates"]["sorted"]
    for gate in sorted_gates:
        child_outputs = [child.output for child in gate.children]
        gate.output = gate.fn_op(inputs=tf.stack(child_outputs, axis=0), name=gate.name)

    sorted_tops: List[Node] = presorted_nodes["nodes"]["tops"]
    outputs = tf.stack([top.output for top in sorted_tops], axis=0)
    return outputs

def quantify(xml_file_path: str, num_batches: int, batch_size: int, sample_size: int):

    # Check if the XML file exists
    if not os.path.isfile(xml_file_path):
        raise FileNotFoundError(f"XML file not found: {xml_file_path}")

    # Parse the XML and build the node graph
    nodes_dict = parse_fault_tree(xml_file_path)
    sorted_nodes = topological_sort(nodes_dict)
    presorted_nodes = presort_event_nodes(sorted_nodes=sorted_nodes)

    rng = tf.random.Generator.from_non_deterministic_state()

    @tf.function(jit_compile=False)
    def run(nodes, num_batches_: int, batch_size_: int, sample_size_: int) -> None:
        for _ in tf.range(num_batches_):
            graph_outputs = build_tf_graph(presorted_nodes=nodes, rng=rng, batch_size=batch_size_, sample_size=sample_size_)
            pop_counts = tf.raw_ops.PopulationCount(x=graph_outputs)
            one_bits = tf.reduce_sum(input_tensor=tf.cast(x=pop_counts, dtype=tf.float64), axis=None)
            mean = one_bits/(tf.reduce_prod(tf.cast(graph_outputs.shape, dtype=tf.float64)) * graph_outputs.dtype.size * 8.0)
            tf.print(mean)

    run(nodes=presorted_nodes, num_batches_=num_batches, batch_size_=batch_size, sample_size_=sample_size)

def setup_env_vars():
    TF_XLA_FLAGS = "--tf_xla_enable_lazy_compilation=false --tf_xla_auto_jit=2"
    TF_XLA_FLAGS += " --tf_mlir_enable_mlir_bridge=true --tf_mlir_enable_convert_control_to_data_outputs_pass=true --tf_mlir_enable_multiple_local_cpu_devices=true"
    TF_XLA_FLAGS += " --tf_xla_deterministic_cluster_names=true --tf_xla_disable_strict_signature_checks=true"
    TF_XLA_FLAGS += " --tf_xla_persistent_cache_directory='./xla/cache/' --tf_xla_persistent_cache_read_only=false"
    os.environ["TF_XLA_FLAGS"] = TF_XLA_FLAGS
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    sys.setrecursionlimit(100000)

if __name__ == '__main__':
    setup_env_vars()

    file_path = "../../tests/fixtures/synthetic-openpsa/ft-openpsa-800Gates.xml"
    num_batches_ = 10
    batch_size_ = 2
    sample_size_ = 2 ** 10
    quantify(xml_file_path=file_path, num_batches=num_batches_, batch_size=batch_size_, sample_size=sample_size_)
