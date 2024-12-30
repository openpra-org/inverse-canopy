import tensorflow as tf

def bitwise_nary_op(bitwise_op, inputs, name: str):
    """
    Efficiently applies the n-ary bitwise op across the specified axis.

    Args:
        bitwise_op (function): The bitwise reduction over the input tensor across the num_events dimension. can be one of `tf.bitwise.bitwise_or`, tf.bitwise.bitwise_and`, `tf.bitwise.bitwise_xor`
        inputs (tf.Tensor): Input tensor with shape (num_events, batch_size, sample_size).
        name (str): A name for this operation

    Returns:
        tf.Tensor: Output tensor with shape [batch_size, sample_size] after reducing the num_events dimension via bitwise op.
    """
    result = tf.foldl(
        fn=bitwise_op,
        elems=inputs,
        initializer=inputs[0, :, :],
        parallel_iterations=(inputs.shape[0]),
        swap_memory=True,
        name=name,
    )
    return result

def bitwise_and(inputs, name: str = "and"):
    return bitwise_nary_op(bitwise_op=tf.bitwise.bitwise_and, inputs=inputs, name=name)

def bitwise_or(inputs, name: str = "or"):
    return bitwise_nary_op(bitwise_op=tf.bitwise.bitwise_or, inputs=inputs, name=name)

def bitwise_xor(inputs, name: str = "xor"):
    return bitwise_nary_op(bitwise_op=tf.bitwise.bitwise_xor, inputs=inputs, name=name)

def bitwise_not(inputs, name: str = "not"):
    return tf.bitwise.invert(x=inputs, name=name)

def bitwise_nand(inputs, name: str = "nand"):
    return bitwise_not(bitwise_and(inputs=inputs,name=name))

def bitwise_nor(inputs, name: str = "nor"):
    return bitwise_not(bitwise_or(inputs=inputs,name=name))

def bitwise_xnor(inputs, name: str = "xnor"):
    return bitwise_not(bitwise_xor(inputs=inputs,name=name))