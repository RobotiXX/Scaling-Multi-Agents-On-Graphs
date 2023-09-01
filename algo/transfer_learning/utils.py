import pickle
import json

def save_to_pickle_file(G_data, file_name):
    # Save graph to a pickle file
    with open('../../data/'+file_name, 'wb') as f:
        pickle.dump(G_data, f)

def load_from_pickle_file(file_name):
    # To read graph back from pickle file
    with open('../../data/'+file_name, 'rb') as f:
        loaded_G = pickle.load(f)
    return loaded_G

def load_graph_from_json(args):
    # To read graph back from json file
    if args.graph_info_file is not  None:
        with open(args.graph_info_file, 'r') as f:
            loaded_graph_info = json.load(f)
        # Converting the keys for 'nodes' back to integers
        loaded_nodes = {int(k): (v,) for k, v in loaded_graph_info['nodes'].items()}

        # Converting the keys for 'edges' back to tuples of integers
        loaded_edges = {tuple(map(int, k.strip('()').split(','))): v for k, v in loaded_graph_info['edges'].items()}
        return loaded_nodes, loaded_edges
    raise ValueError("No graph info file found")