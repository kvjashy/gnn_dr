
"""
    Some helper functions to parse the mujoco xml template files

    @author:
        Tobias Schmidt, Hannes Stark, modified from the code of Tingwu Wang.
"""


import os
import logging
import numpy as np
import pybullet_data

from pathlib import Path
from enum import IntEnum, Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from bs4 import BeautifulSoup

from graph_util.mujoco_parser_settings import ALLOWED_NODE_TYPES, EDGE_TYPES, ControllerType, RootRelationOption
from graph_util.mujoco_parser_settings import XML_DICT as PYBULLET_XML_DICT
from graph_util.mujoco_parser_nervenet import XML_DICT as NERVENET_XML_DICT

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ["parse_mujoco_graph"]

XML_ASSETS_DIR = Path(pybullet_data.getDataPath()) / "mjcf"


def parse_mujoco_graph(task_name: str = None,
                       xml_name: str = None,
                       xml_assets_path: Path = None,
                       allowed_node_types: List[str] = ALLOWED_NODE_TYPES,
                       use_sibling_relations: bool = True,
                       root_relation_option: RootRelationOption = RootRelationOption.BODY):
    '''
    TODO: add documentation

    Parameters:
        "task_name":
            The name of the task to parse the graph structure from.
            Takes priority over xml_name.
        "xml_name":
            The name of the xml file to parse the graph structure from.
            Either xml_name or task_name must be specified.
        "xml_assets_path":
            Specifies in which directory to look for the mujoco (mjcf) xml file.
            If none, default will be set to the pybullet data path.
        "use_sibling_relations":
            Whether to use sibling relations between nodes to build the relation (adjacency) matrix  
            or to only use parent-child relationships
        "root_relation_option":
                Specifies which other nodes the root node should additionally be connected to:
                    - RootRelationOption.NONE:
                        No other nodes 
                    - RootRelationOption.BODY:
                        All nodes of type "body" are connected to root
                    - RootRelationOption.ALL:
                        All nodes are connected to root

    Returns:
        A dictionary containing details about the parsed graph structure:
            "tree": dict
                A dictionary representation of the parsed XML
            "relation_matrix": ndarray of shape (num_nodes, num_nodes)
                A representation of the adjacency matrix for the parsed graph.
                Non-Zero entries are edge conections of different types as defined by EDGE_TYPES
            "node_type_dict": dict
                Keys: The names of the node types
                Values: The list of the node ids that are of this node type
            "output_type_dict": dict
                Specifies which outputs should use the same controller network
                Keys: The name of the control group (e.g. "hip", "ankle")
                Values: The list of the node ids that are of this motor type
            "output_list": list
                The list of node ids that correspond to each of the motors.
                The order exactly matches the motor order specified in the xml file.
            "input_dict": dict
                Keys: node ids
                Value: list of ids in observation vector that belong to the node
            "node_parameters"
    '''

    if task_name is not None:  # task_name takes priority
        if task_name in NERVENET_XML_DICT:
            xml_name = NERVENET_XML_DICT[task_name]
        elif task_name in PYBULLET_XML_DICT:
            xml_name = PYBULLET_XML_DICT[task_name]
        else:
            raise NotImplementedError(f"No task named {task_name} defined.")

    assert xml_name is not None, "Either task_name or xml_name must be given."

    if xml_assets_path is None:
        xml_assets_path = XML_ASSETS_DIR

    xml_path = xml_assets_path / xml_name

    with open(str(xml_path), "r") as xml_file:
        xml_soup = BeautifulSoup(xml_file.read(), "xml")

    tree = __extract_tree(xml_soup)

    relation_matrix = __build_relation_matrix(tree,
                                              use_sibling_relations=use_sibling_relations,
                                              root_relation_option=root_relation_option)

    # group nodes by node type
    node_type_dict = {node_type: [node["id"]
                                  for node in tree if node["type"] == node_type]
                      for node_type in allowed_node_types}

    output_type_dict, output_list = __get_output_mapping(tree)

    # TODO
    input_dict = {}
    debug_info = {}
    node_parameters = {}
    para_size_dict = {}

    return dict(tree=tree,
                relation_matrix=relation_matrix,
                node_type_dict=node_type_dict,
                output_type_dict=output_type_dict,
                output_list=output_list,
                input_dict=input_dict,
                debug_info=debug_info,
                node_parameters=node_parameters,
                para_size_dict=para_size_dict,
                num_nodes=len(tree))


def __extract_tree(xml_soup: BeautifulSoup,
                   allowed_node_types: List[str] = ALLOWED_NODE_TYPES):
    '''
    TODO: Add docstring
    '''

    motor_names = __get_motor_names(xml_soup)
    robot_body_soup = xml_soup.find("worldbody").find("body")

    root_joints = robot_body_soup.find_all('joint', recursive=False)

    root_node = {"type": "root",
                 "is_output_node": False,
                 "name": "root_mujocoroot",
                 "neighbour": [],
                 "id": 0,
                 "parent": 0,
                 "info": robot_body_soup.attrs,
                 "attached_joint_name": ["joint_" + j["name"] for j in root_joints if j["name"] not in motor_names],
                 "attached_joint_info": [],
                 "motor_names": motor_names
                 }

    # TODO: Add observation size information

    tree = list(__unpack_node(robot_body_soup,
                              current_tree={0: root_node},
                              parent_id=0,
                              motor_names=motor_names).values())

    # absorb joints that are directly attached to the root node into the root node
    # not realy sure why we do this, but that's how the original NerveNet Implementation does it...
    tree[0]["attached_joint_info"] = [node
                                      for node in tree
                                      if node["name"] in tree[0]["attached_joint_name"]]
    tree = [node
            for node in tree
            if node["name"] not in tree[0]["attached_joint_name"]]

    # after absorbing the root joints we need to adjust the indicies accordingly
    index_offset = len(tree[0]["attached_joint_name"])
    for i in range(len(tree)):
        # apply only for indicies that came after the ones that were removed
        if tree[i]["id"] - index_offset > 0:
            tree[i]["id"] = tree[i]["id"] - index_offset
        if tree[i]["parent"] - index_offset > 0:
            tree[i]["parent"] = tree[i]["parent"] - index_offset

    return tree


def __unpack_node(node: BeautifulSoup,
                  current_tree: dict,
                  parent_id: int,
                  motor_names: List[str],
                  allowed_node_types: List[str] = ALLOWED_NODE_TYPES) -> dict:
    '''
    This function is used to recursively unpack the xml graph structure of a given node.

    Parameters:
        node: 
            The root node from which to unpack the underlying xml tree into a dictionary.
        current_tree:
            The dictionary represenation of the tree that has already been unpacked. 
            - Required to determin the new id to use for the current node.
            - Will be updated with the root node and its decendent nodes.
        parent_id: 
            The id of the parent node for the current node.
        motor_names: 
            The list of joint names that are supposed be output_nodes.
        allowed_node_types: 
            The list of tag names that should be extracted as decendent nodes

    Returns:
        A dictionary representation of the xml tree rooted at the given node
        with "id" and "parent_id" references to encode the relationship structure.
    '''
    id = max(current_tree.keys()) + 1
    node_type = node.name

    current_tree[id] = {
        "type": node_type,
        "is_output_node": node["name"] in motor_names,
        "raw_name": node["name"],
        "name": node_type + "_" + node["name"],
        "id": id,
        "parent": parent_id,  # to be set later
        "info": node.attrs
    }

    child_soups = [child
                   for allowed_type in allowed_node_types
                   for child in node.find_all(allowed_type, recursive=False)
                   ]

    for child in child_soups:
        current_tree.update(__unpack_node(child,
                                          current_tree=current_tree,
                                          parent_id=id,
                                          motor_names=motor_names,
                                          allowed_node_types=allowed_node_types))

    return current_tree


def __get_motor_names(xml_soup: BeautifulSoup) -> List[str]:
    motors = xml_soup.find('actuator').find_all('motor')
    name_list = [motor['joint'] for motor in motors]
    return name_list


def __build_relation_matrix(tree: List[dict],
                            use_sibling_relations: bool = True,
                            root_relation_option: RootRelationOption = RootRelationOption.BODY) -> np.ndarray:
    '''
    TODO better docstring

    Parameters:
        "tree": 
            The dictionary representation of the parsed XML to build the relation matrix from.
        "use_sibling_relations":
            Whether to use sibling relations between nodes to build the relation (adjacency) matrix  
            or to only use parent-child relationships
        "root_relation_option":
            The root node is an abstract node and not part of the kinematic tree described by the XML structure.
            Therefore we need to define which nodes it is in relation/connected to.
            At a minimum it should have the same relations as the main body (e.g. "torso") node does.
            With this option we can additionally specify which other nodes it should also be connected to:
                - RootRelationOption.NONE:
                    No other nodes 
                - RootRelationOption.BODY:
                    All nodes of type "body" are connected to root
                - RootRelationOption.ALL:
                    All nodes are connected to root
    Returns:
        "relation_matrix": ndarray of shape (num_nodes, num_nodes)
                A representation of the adjacency matrix for the parsed graph.
                Non-Zero entries are edge conections of different types as defined by EDGE_TYPES

    '''
    num_node = len(tree)
    relation_matrix = np.zeros([num_node, num_node], dtype=np.int)

    # nodes with outgoing edge
    for node_out in tree:
        # nodes with incoming edge
        for node_in in tree:
            if node_in["parent"] == node_out["id"]:
                # direction out -> in
                relation_matrix[node_out["id"]][node_in["id"]] = EDGE_TYPES[(
                    node_out["type"], node_in["type"])]
                # direction in -> out
                relation_matrix[node_in["id"]][node_out["id"]] = EDGE_TYPES[(
                    node_in["type"], node_out["type"])]

    # always connect the root to its grand-children, but not to its children
    # for whatever reasons... again that's just how the reference implementation works
    root_children_ids = [node["id"]
                         for node in tree
                         if node["parent"] == 0
                         and node["id"] != 0]
    # we only care about the children of type body
    root_grandchildren = [node
                          for node in tree
                          if node["parent"] in root_children_ids
                          and node["type"] == "body"]

    # disconnect root with its children
    for child_node_id in root_children_ids:
        relation_matrix[child_node_id][0] = 0
        relation_matrix[0][child_node_id] = 0

    # connect root with its grand-children
    for grandchild_node in root_grandchildren:
        relation_matrix[grandchild_node["id"]][0] = EDGE_TYPES[(
            grandchild_node["type"], "root")]
        relation_matrix[0][grandchild_node["id"]] = EDGE_TYPES[(
            "root", grandchild_node["type"])]

    if use_sibling_relations:
        sibling_pairs = [(node_1, node_2)
                         for node_1 in tree
                         for node_2 in tree
                         if node_1["parent"] == node_2["parent"]
                         and node_1["id"] != node_2["id"]
                         and node_1["type"] != "root"
                         and node_2["type"] != "root"]
        for node_1, node_2 in sibling_pairs:
            relation_matrix[node_1["id"]][node_2["id"]] = EDGE_TYPES[(
                node_1["type"], node_2["type"])]
            relation_matrix[node_2["id"]][node_1["id"]] = EDGE_TYPES[(
                node_2["type"], node_1["type"])]

    if root_relation_option != RootRelationOption.NONE:
        if root_relation_option == RootRelationOption.ALL:
            root_relation_nodes = tree
        elif root_relation_option == RootRelationOption.BODY:
            root_relation_nodes = [node
                                   for node in tree
                                   if node["type"] == "body"]
        else:
            raise NotImplementedError("Unknown root relation option")

        for node in root_relation_nodes:
            if node["type"] != "root":
                relation_matrix[node["id"]][0] = EDGE_TYPES[(
                    node["type"], "root")]
                relation_matrix[0][node["id"]] = EDGE_TYPES[(
                    "root", node["type"])]

    return relation_matrix


def __get_output_mapping(tree: List[dict], controller_type: ControllerType = ControllerType.SHARED) -> Tuple[dict, List[int]]:
    '''
    Parameters:
        "tree": 
            The dictionary representation of the parsed XML to extract the output mapping from.
        "controller_type": 
            Wang et al. propose different sharing settings for the controller networks that generate
            the action from the final hidden presentation of actuator/motor/output nodes.
                - shared
                    The same controller network for the same/similar type of motors TODO: specify what exactly is "similiar"
                - seperate
                    Different controller networks for every output
                - unified:
                    The same controller network for all outputs
    Returns:
        "output_type_dict": dict
            A mapping of output nodes into controll groups. 
            Nodes of the same controll group are supposed to share the same controller network.
            Keys: The identifier of the group
            Values: The list of nodes of this group
        "output_list": list[str]
            The list of all output nodes in the order
    '''

    # we need to use the exact same order as motors are defined in the xml
    # --> names in the correct order are in root node
    # --> indexing first improves sorting afterwards
    motor_index_map = {v: i for i, v in enumerate(tree[0]["motor_names"])}

    output_nodes = {node["raw_name"]: node
                    for node in tree if node["is_output_node"]}
    output_nodes = dict(sorted(output_nodes.items(),
                               key=lambda pair: motor_index_map[pair[0]]))

    output_list = [node["id"] for node in output_nodes.values()]

    # TODO currently we just type/group together based on the name prefix....
    # either we need to document this as "feature" or find a better way to do this
    joint_types = set([node_name.split("_")[0]
                       for node_name in output_nodes.keys()])

    if controller_type == ControllerType.SHARED:
        output_type_dict = {
            joint_type: [node["id"]
                         for node_name, node in output_nodes.items() if joint_type in node_name]
            for joint_type in joint_types
        }
    elif controller_type == ControllerType.SEPERATE:
        output_type_dict = {node["raw_name"]: node["id"]
                            for node in output_nodes}
    elif controller_type == ControllerType.UNIFIED:
        output_type_dict['unified'] = output_list
    else:
        raise NotImplementedError(
            "Unknown controller type: %s" % controller_type)

    return output_type_dict, output_list


if __name__ == "__main__":
    assert False