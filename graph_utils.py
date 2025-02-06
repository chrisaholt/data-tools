
import numpy as np
from typing import List

class Node:
    """Vertices of a graph"""
    def __init__(self, description: str):
        self.description = description
    
    def __str__(self):
        return self.description
    
    def __repr__(self):
        return self.description

class Graph:
    """Nodes with edges"""
    def __init__(self):
        self.vertices = set()
        self.edges = {}

    def add_vertex(self, node: Node):
        self.vertices.add(node)

    def add_edge(self, start: Node, end: Node):
        assert start in self.vertices, "Start node not found."
        assert end in self.vertices, "End node not found."
        
        if start not in self.edges:
            self.edges[start] = set()
        self.edges[start].add(end)
    
    def neighbors(self, start: Node):
        if start not in self.edges:
            return []
        return self.edges[start]
    

def shortest_path_dijkstra(
        graph: Graph,
        start: Node,
        end: Node,
        debug: bool=False,
    ) -> List[Node]:
    """Find the shortest path between two nodes in a graph"""

    # A dictionary that stores the shortest path from the start node to all other nodes
    shortest_paths_from_start = {start: [start]}
    
    # A set of nodes whose shortest path from start have been finalized
    finalized_nodes = set()
    finalized_nodes.add(start)
    last_added_node = start

    # Keep a list of visited nodes.
    visited_nodes = set()
    visited_nodes.add(start)

    while end not in shortest_paths_from_start:
        if debug:
            print()
            print(f"***finalized_nodes: {finalized_nodes}")
            print(f"***visited_nodes: {visited_nodes}")

        # Look at all the neighbors. Compare the path length from start.
        neighbors = graph.neighbors(last_added_node)
        for neighbor in neighbors:
            if debug:
                print(f"***neighbors: {neighbors}")
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                new_path = shortest_paths_from_start[last_added_node] + [neighbor]
                if neighbor not in shortest_paths_from_start:
                    shortest_paths_from_start[neighbor] = new_path
                else:
                    if len(new_path) < len(shortest_paths_from_start[neighbor]):
                        shortest_paths_from_start[neighbor] = new_path

        # Find the shortest path among all the paths found, and label that final.
        min_path_length = np.inf
        min_path_node = None
        for node in shortest_paths_from_start:
            if node not in finalized_nodes:
                path_length = len(shortest_paths_from_start[node])
                if path_length < min_path_length:
                    min_path_node = node
                    min_path_length = path_length

        if min_path_node is None:
            # There is no path to end node.
            return None
        last_added_node = min_path_node
        finalized_nodes.add(min_path_node)

    return shortest_paths_from_start[end]