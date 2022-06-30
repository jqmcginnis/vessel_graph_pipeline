import os
import os.path as osp
import torch
import numpy as np
import scipy
import pandas as pd
import argparse
import os
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_gz, extract_tar, extract_zip)
from torch_geometric.data.makedirs import makedirs
from torch_geometric.utils import to_undirected, remove_isolated_nodes, remove_self_loops
from torch_sparse import coalesce

parser = argparse.ArgumentParser(description='display graph features and summary.')
parser.add_argument('-ds','--dataset',help='Specify the dataset you want to select', required=True)
parser.add_argument('-s','--splitting_strategy',help='Specify the dataset you want to select', required=True)
parser.add_argument('-l','--min_vessel_length', type=float, default=5.0, help='Minimum vessel length')
parser.add_argument('-min', '--min_cycle_length',type=int, help='Specify the minimum required number of nodes.', default=3)
parser.add_argument('-max', '--max_cycle_length', type=int, help='Specify the minimum required number of nodes.', default=15)

args = parser.parse_args()
selected_dataset = args.dataset

# import PyTorch libs
from link_dataset import LinkVesselGraph

def main():

    dataset = LinkVesselGraph(root='data', name=selected_dataset, splitting_strategy=args.splitting_strategy,
                              min_vessel_length=args.min_vessel_length,
                              use_edge_attr=True, use_atlas=True)

    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    print(dataset.name)

    data = dataset[0]  # Get the first graph object.
    print(data)
    print('==============================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is Undirected: {data.is_undirected()}')

    print(data.edge_index_undirected)
    print(data.edge_index)


    complete_graph = Data(x = data.x, edge_index=data.edge_index_undirected, edge_attr=data.edge_attr_undirected)

    #x = data.pos.cpu()
    #edge_attr_undirected = data.edge_attr_undirected

    # convert to networkx for computations
    graph = to_networkx(data=complete_graph,
                        node_attrs=["x"],
                        edge_attrs=["edge_attr"],
                        to_undirected=True,
                        remove_self_loops=False)#, node_attrs = data.x,edge_attrs=data.edge_attr_undirected,to_undirected=True, remove_self_loops=False)

    #print(graph.nodes(data=True))
    #print(graph.edges(data=True))

    ## compute closed loops for undirected graphs
    cycles = nx.cycle_basis(graph)
    nodes = data.pos.cpu().detach().numpy()

    # restrict cycles to <=10
    valid_indices = []

    # toss out the ones we don't want
    for cycle_idx, cycle in enumerate(cycles):
        if ((len(cycle) >= args.min_cycle_length) and (len(cycle) <= args.max_cycle_length)):
            valid_indices.append(cycle)

    cycles = valid_indices

    # declare bounding_box_array
    bounding_boxes = np.zeros((len(cycles),6))
    loop_number = []
    edge_count = []

    for cycle_idx, cycle in enumerate(cycles):
        if len(cycle) <= args.max_cycle_length:
            loop_number.append(cycle_idx)
            edge_count.append(len(cycle))
            # assemble the node list
            # x, y, z
            local_bounding_box = np.zeros((len(cycle),3),dtype=float)
            for idx,x in enumerate(cycle):
                local_bounding_box[idx,:] = nodes[int(x),:]

                bounding_boxes[cycle_idx, 0] = np.min(local_bounding_box[:,0])
                bounding_boxes[cycle_idx, 1] = np.max(local_bounding_box[:, 0])
                bounding_boxes[cycle_idx, 2] = np.min(local_bounding_box[:, 1])
                bounding_boxes[cycle_idx, 3] = np.max(local_bounding_box[:, 1])
                bounding_boxes[cycle_idx, 4] = np.min(local_bounding_box[:, 2])
                bounding_boxes[cycle_idx, 5] = np.max(local_bounding_box[:, 2])

    cycle_length = []
    vessel_length = []
    vessel_distance = []

    index_len = data.edge_attr_keys.index("length")
    index_dist = data.edge_attr_keys.index("distance")

    for cycle in cycles:
        i = 0
        vl = 0
        dist = 0
        while i < len(cycle)-1:
            # hop from edge to edge
            vl += graph.get_edge_data(cycle[i],cycle[i+1])['edge_attr'][index_len]
            dist += graph.get_edge_data(cycle[i],cycle[i+1])['edge_attr'][index_dist]
            i = i+1
        # finish computation by closing the loop again [distance and length from first to last element]
        vl += graph.get_edge_data(cycle[-1], cycle[0])['edge_attr'][index_len]
        dist += graph.get_edge_data(cycle[-1], cycle[0])['edge_attr'][index_dist]
        vessel_length.append(vl)
        vessel_distance.append(dist)

    d = {'loop_nr':loop_number,
         'edge_count': edge_count,
         'xmin':bounding_boxes[:,0],
         'xmax': bounding_boxes[:, 1],
         'ymin': bounding_boxes[:, 2],
         'ymax': bounding_boxes[:, 3],
         'zmin': bounding_boxes[:, 4],
         'zmax': bounding_boxes[:, 5],
         'vessel_length': vessel_length,
         'vessel_dist': vessel_distance,
         'cycle_elements': cycles,
         }
    df = pd.DataFrame(data=d)
    identifier = f"{dataset.name}_num_closed_loops_edge_len.csv"
    df.to_csv(identifier) 


if __name__ == "__main__":
        main()



