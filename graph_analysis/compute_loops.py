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

args = parser.parse_args()
selected_dataset = args.dataset

# import PyTorch libs
from link_dataset import LinkVesselGraph

def main():

    dataset = LinkVesselGraph(root='data', name=selected_dataset, splitting_strategy=args.splitting_strategy, use_edge_attr=True, use_atlas=True)

    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

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

    print(data.train_pos_edge_index.shape)
    print(data.test_pos_edge_index.shape)
    print(data.val_pos_edge_index.shape)

    print(data.train_neg_edge_index.shape)
    print(data.test_neg_edge_index.shape)
    print(data.val_neg_edge_index.shape)

    print('==============================================================')

    ## convert to networkx graph to do some graph analysis

    complete_graph = Data(x = data.x, edge_index=data.edge_index_undirected, edge_attr=data.edge_attr_undirected)
    print(complete_graph)

    graph = to_networkx(data=complete_graph,to_undirected=True, remove_self_loops=False)#, node_attrs = data.x,edge_attrs=data.edge_attr_undirected,to_undirected=True, remove_self_loops=False)
    print(graph)
    ## compute closed loops for undirected graphs
    cycles = nx.cycle_basis(graph)
    print(cycles)
    print("Nuber of closed loop",len(cycles))



if __name__ == "__main__":
        main()



