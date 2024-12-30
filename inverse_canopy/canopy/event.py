from typing import List, Optional
import tensorflow as tf

class Node:
    def __init__(self, name: str, is_top: bool = False, level: int = -1):
        self.name = name
        self.output: Optional[tf.Tensor] = None  # To hold the TensorFlow tensor corresponding to this node
        self.is_top: bool = is_top
        self.level: int = level


class Gate(Node):
    def __init__(self, name: str, gate_type: str, is_top: bool = True):
        super().__init__(name=name, is_top=is_top)
        self.gate_type = gate_type
        self.children: List[Gate | BasicEvent] = []
        self.fn_op = None


class BasicEvent(Node):
    def __init__(self, name: str, probability: float):
        super().__init__(name=name, is_top=False, level=0)
        self.probability = probability

    @property
    def children(self):
        return []