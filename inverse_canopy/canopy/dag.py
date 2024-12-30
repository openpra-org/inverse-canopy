from collections import defaultdict
from typing import Dict, List, Any

from inverse_canopy.canopy.event import Node, Gate, BasicEvent
from inverse_canopy.canopy.ops.bitwise import bitwise_and, bitwise_or, bitwise_xor, bitwise_nand, bitwise_xnor, bitwise_not, bitwise_nor


# Function to perform a topological sort
def topological_sort(nodes_dict: Dict[str, Gate | BasicEvent]) -> List[Gate | BasicEvent]:
    visited = set()
    sorted_nodes: List[Gate | BasicEvent] = []

    def dfs(node: Gate | BasicEvent):
        if node.name in visited:
            return
        visited.add(node.name)
        if isinstance(node, Gate):
            for child in node.children:
                dfs(child)
        sorted_nodes.append(node)

    # Identify top nodes (nodes marked as 'is_top')
    top_nodes = [node for node in nodes_dict.values() if getattr(node, 'is_top', False)]
    if not top_nodes:
        # If no top nodes are marked, treat nodes without parent as top nodes
        child_names = {child.name for node in nodes_dict.values() for child in node.children}
        top_nodes = [node for node in nodes_dict.values() if isinstance(node, Gate) and node.name not in child_names]

    # Start DFS from the top nodes
    for node in top_nodes:
        dfs(node)

    # Check for cycles (if not all nodes are visited)
    if len(visited) != len(nodes_dict):
        undefined_nodes = set(nodes_dict.keys()) - visited
        raise ValueError(f"Cycles detected or undefined nodes: {undefined_nodes}")

    return list(reversed(sorted_nodes))  # Reverse to get the correct order

OpMap: Dict[str, Any] = {
    "and": bitwise_and,
    "or": bitwise_or,
    "xor": bitwise_xor,
    "nor": bitwise_nor,
    "xnor": bitwise_xnor,
    "not": bitwise_not,
}

def op_for_gate_type(gate_type: str):
    if gate_type == "and":
        return bitwise_and
    elif gate_type == "or":
        return bitwise_or
    elif gate_type == "not":
        return bitwise_not
    elif gate_type == "xor":
        return bitwise_xor
    elif gate_type == "xnor":
        return bitwise_xnor
    elif gate_type == "nand":
        return bitwise_nand
    else:
        raise ValueError(f"Unknown gate: {gate_type}")

def presort_event_nodes(sorted_nodes: List[Node]):
    # Initialize dictionaries to hold outputs
    # node_outputs: Dict[str, tf.Tensor] = {}
    basic_event_probs: List[float] = []
    basic_event_probability_indices: Dict[str, int] = {}

    sorted_basic_events: List[BasicEvent] = []
    sorted_gates: List[Gate] = []

    sorted_tops: List[Node] = []

    gates_by_level = defaultdict(list)

    # First pass: collect basic event probabilities, and set gate op type
    for node in reversed(sorted_nodes):
        if isinstance(node, BasicEvent):
            sorted_basic_events.append(node)
            if node.name not in basic_event_probability_indices:
                basic_event_probability_indices[node.name] = len(basic_event_probs)
                basic_event_probs.append(node.probability)
        elif isinstance(node, Gate):
            if len(node.children) == 0:
                print("empty node!", node.name)
            node.level = 1 + max(child.level for child in node.children)
            node.fn_op = OpMap.get(node.gate_type)
            sorted_gates.append(node)
            # finally, add this to node it's level
            gates_by_level[node.level].append(node)
        if node.is_top:
            sorted_tops.append(node)


    presorted_nodes = {
        "nodes": {
            "sorted": reversed(sorted_nodes),
            "tops": sorted_tops,
            "by_level": gates_by_level,
        },
        "gates": {
            "sorted": sorted_gates,
            "by_level": gates_by_level,
        },
        "basic_events": {
            "sorted": sorted_basic_events,
            "probabilities": basic_event_probs,
            "probability_indices": basic_event_probability_indices,
        },
    }

    print(f"tops: {len(sorted_tops)}, gates: {len(sorted_gates)}, basic events: {len(sorted_basic_events)}")
    print(f"total_levels:{len(gates_by_level.keys())}")
    for key, value in gates_by_level.items():
        print(f"level {key}: {len(value)}, num_inputs: {sum([len(node.children) for node in value])}, {[node.name for node in value]}")
    return presorted_nodes
