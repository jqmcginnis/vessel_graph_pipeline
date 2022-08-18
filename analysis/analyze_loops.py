import os.path as osp
import torch
import numpy as np
import pandas as pd
import os
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes, remove_self_loops
from tqdm import tqdm

def analyze_loops(node_list, edge_list, min_vessel_length, min_cycle_length, max_cycle_length, file_path):

    df_nodes = pd.read_csv(node_list,sep=';')
    df_edges = pd.read_csv(edge_list,sep=';')

    data = Data()
    data.node_attr_keys = ['pos_x','pos_y','pos_z']

    # Node feature matrix with shape [num_nodes, num_node_features]
    data.x = torch.from_numpy(np.array(df_nodes[data.node_attr_keys].to_numpy()))

    # Node position matrix with shape [num_nodes, num_dimensions]
    data.pos = torch.from_numpy(np.array( df_nodes[['pos_x', 'pos_y', 'pos_z']].to_numpy())) # coordinates

    data.edge_attr_keys = ['length','distance','avgRadiusAvg','roundnessAvg','curveness']

    edges = np.column_stack((np.array(df_edges[['node1id']]),np.array(df_edges[['node2id']])))
    edge_features = np.array(df_edges[data.edge_attr_keys].to_numpy())

    # filter minimum edge length
    idx_length = data.edge_attr_keys.index("length")
    indices = np.squeeze(np.argwhere(edge_features[:,idx_length]>=min_vessel_length))

    edge_features = edge_features[indices]
    edges = edges[indices]

    data.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data.edge_attr = torch.from_numpy(np.array(edge_features))

    # remove self loops
    data.edge_index , data.edge_attr = remove_self_loops(data.edge_index,data.edge_attr)

    # filter out isolated nodes
    data.edge_index, data.edge_attr , node_mask = remove_isolated_nodes(edge_index=data.edge_index,edge_attr = data.edge_attr,num_nodes=data.num_nodes)
    data.x = data.x[node_mask]
    data.pos = data.pos[node_mask]

    # problem does not order them as I would like
    # data.edge_index, data.edge_attr = to_undirected(edge_index=data.edge_index,edge_attr = data.edge_attr,num_nodes=data.num_nodes,reduce="add") # add attribute

    edge_array = np.ones((2,int(2*data.edge_attr.shape[0])))
    edge_attr_array = np.ones((2*data.edge_attr.shape[0],data.edge_attr.shape[1]))

    for i in range(0,data.edge_attr.shape[0]):
        edge_array[0,2*i] = int(data.edge_index[0,i])
        edge_array[1, 2*i] = int(data.edge_index[1, i])
        edge_array[1, 2*i+1] = int(data.edge_index[0,i])
        edge_array[0, 2*i+1] = int(data.edge_index[1, i])
        edge_attr_array[2*i,:] = np.array(data.edge_attr[i,:])
        edge_attr_array[2*i+1,:] = np.array(data.edge_attr[i,:])

    # includes all edges (train+test+val) in both drections
    data.edge_index_undirected = torch.tensor(edge_array, dtype=torch.long)
    data.edge_attr_undirected = torch.tensor(edge_attr_array, dtype=torch.float)

    complete_graph = Data(x = data.x, edge_index=data.edge_index_undirected, edge_attr=data.edge_attr_undirected)

    # convert to networkx for computations
    graph = to_networkx(data=complete_graph,
                        node_attrs=["x"],
                        edge_attrs=["edge_attr"],
                        to_undirected=True,
                        remove_self_loops=False)

    ## compute closed loops for undirected graphs
    cycles = nx.cycle_basis(graph)
    nodes = data.pos.cpu().detach().numpy()

    # restrict cycles to <=10
    valid_indices = []

    # toss out the ones we don't want
    for cycle_idx, cycle in enumerate(tqdm(cycles)):
        if ((len(cycle) >= min_cycle_length) and (len(cycle) <= max_cycle_length)):
            valid_indices.append(cycle)

    cycles = valid_indices

    # declare bounding_box_array
    bounding_boxes = np.zeros((len(cycles),6))
    loop_number = []
    edge_count = []

    for cycle_idx, cycle in enumerate(tqdm(cycles)):
        if len(cycle) <= max_cycle_length:
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

    vessel_length = []
    vessel_distance = []

    index_len = data.edge_attr_keys.index("length")
    index_dist = data.edge_attr_keys.index("distance")

    for cycle in tqdm(cycles):
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

    df.to_csv(file_path) 



