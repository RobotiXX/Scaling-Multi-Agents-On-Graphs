from typing import Dict, Tuple
import networkx as nx
import random
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, nodes: Dict[int, Tuple[int]], edges: Dict[Tuple[int, int], float]):
        self.graph = nx.Graph()
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)
        self.start_goal = 0
        self.end_goal = len(self.graph.nodes) - 1

    def __contains__(self, node):
        return node in self.graph.nodes

    def get_neighbors(self, node):
        return list(self.graph.neighbors(node))
    
    def get_fixed_start_goal(self):
        return self.start_goal
    
    def get_fixed_end_goal(self):
        return self.end_goal

    def get_random_start_goal(self):
        nodes = list(self.graph.nodes)
        random_index = random.randint(0, len(nodes) - 1)
        random_node = nodes[random_index]
        self.start_goal = random_node
        return self.start_goal
    
    def get_random_end_goal(self):
        nodes = list(self.graph.nodes)
        while True:
            random_index = random.randint(0, len(nodes) - 1)
            random_node = nodes[random_index]

            if random_node != self.start_goal:
                break
        self.end_goal = random_node
        return self.end_goal
    
    def get_random_node_from_list(self, nodes):
        random_index = random.randint(0, len(nodes) - 1)
        random_node = nodes[random_index]
        return random_node   

    def shortest_path(self, start, goal):
        return nx.shortest_path(self.graph, start, goal)
    
    def shortest_path_length(self, start, goal):
        return nx.shortest_path_length(self.graph, start, goal)
    
    def draw_graph(self):
        nx.draw(self.graph, with_labels=True, font_weight='bold', node_color='skyblue', font_color='black')
        plt.savefig('10_nodes_graph_like_grid.png')
        plt.grid()

    # check density of graph
    def is_graph_sparse_or_dense(self):
        num_edges = self.graph.number_of_edges()
        num_vertices = self.graph.number_of_nodes()
        
        # For a simple graph, the maximum number of edges is "n choose 2"
        max_edges = (num_vertices * (num_vertices - 1)) / 2
        
        # Calculate density
        density = 2 * num_edges / (num_vertices * (num_vertices - 1))
        
        print(f"Density of graph: {density}")
        
        if density > 0.5:
            return "Dense"
        else:
            return "Sparse"