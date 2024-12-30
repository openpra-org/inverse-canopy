from typing import Dict
from lxml import etree
from inverse_canopy.canopy.event import Gate, BasicEvent


def parse_fault_tree(xml_file_path: str) -> Dict[str, Gate | BasicEvent]:
    # Parse the XML file using lxml
    parser = etree.XMLParser(ns_clean=True, recover=True)
    xml_tree = etree.parse(xml_file_path, parser)
    root = xml_tree.getroot()

    # Build nodes and store them in a dictionary
    nodes: Dict[str, Gate | BasicEvent] = {}

    # Parse basic events from <model-data>
    for event_xml in root.xpath("//model-data/define-basic-event"):
        name = event_xml.get("name")
        float_elem = event_xml.find("float")
        probability = float(float_elem.get("value")) if float_elem is not None else 0.0
        event_node = BasicEvent(name=name, probability=probability)
        nodes[name] = event_node

    # parse all gate names and types
    for gate_xml in root.xpath("//define-fault-tree/define-gate"):
        gate_name = gate_xml.get("name")
        gate_type = str(gate_xml[0].tag)

        if gate_type == "not":
            not_gate_xml = gate_xml[0]
            child_gate_type = str(not_gate_xml[0].tag)
            match child_gate_type:
                case "and": gate_type = "nand"
                case "or": gate_type = "nor"
                case "xor": gate_type = "xnor"

        nodes[gate_name] = Gate(name=gate_name, gate_type=gate_type, is_top=True) # set all to top for now

    # Parse fault trees to find top gates
    for ft_xml in root.xpath("//define-fault-tree"):
        first_gate_xml = ft_xml.xpath(".//define-gate")[0]
        first_gate_name = first_gate_xml.get("name")
        first_gate_node: Gate = nodes[first_gate_name]
        first_gate_node.is_top = True # set the first one to true

        # Process gates defined within the fault tree
        for gate_xml in ft_xml.xpath(".//define-gate"):
            gate_name = gate_xml.get("name")
            gate_node: Gate = nodes[gate_name]
            #print(gate_node.is_top, gate_node.name, gate_node.gate_type, len(gate_node.children))
            # Determine gate type and collect children
            gate_xml_to_traverse = gate_xml
            if gate_node.gate_type == "nand" or gate_node.gate_type == "nor" or gate_node.gate_type == "xnor":
                gate_xml_to_traverse = gate_xml[0]
            elif gate_node.gate_type == "basic-event":
                print(gate_node, gate_name, gate_xml)
            elif gate_node.gate_type == "gate": # just a reference to another gate
                print(gate_node, gate_name, gate_xml)
            # There should be only one child element representing the gate type
            for elem in gate_xml_to_traverse:
                # Parse the children of the gate
                for child_elem in elem:
                    child_name = child_elem.get("name")
                    if child_name is not None:
                        child_node = nodes[child_name]
                        if isinstance(child_node, Gate):
                            child_node.is_top = False # child can't be top!
                        gate_node.children.append(child_node)
                break  # Process only the first gate type element

            # If this gate is directly under <define-fault-tree>, mark it as a top node

        break # just parse the first fault tree

    # Return the nodes dictionary
    return nodes

