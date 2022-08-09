import pandas as pd
import numpy as np
import argparse
import os
import argparse
import os
import os.path as osp
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='generate vectors')
parser.add_argument('-i','--input', help='Name of input directory', required=True)

# read the arguments
args = vars(parser.parse_args())
folder = args['input']

subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]

for i in range(len(subfolders)):

    # get the identifier
    id = os.path.basename(os.path.normpath(subfolders[i]))

    print("Processing:")

    print(osp.join(folder,id, f'{id}_nodes.csv'))
    print(osp.join(folder,id, f'{id}_edges.csv'))

    df_nodes = pd.read_csv(osp.join(folder,id, f'{id}_nodes.csv'),sep=';')
    df_edges = pd.read_csv(osp.join(folder,id, f'{id}_edges.csv'),sep=';')

    # debugging
    df_nodes.info(verbose=True)

    # Node feature matrix with shape [num_nodes, num_node_features]
    x = np.array(df_nodes[['degree','isAtSampleBorder']].to_numpy())

    # data.pos: Node position matrix with shape [num_nodes, num_dimensions]
    pos = np.array( df_nodes[['pos_x', 'pos_y', 'pos_z']].to_numpy())
    pos = np.abs(pos)

    print("Pos", pos.shape)

    # https://en.wikipedia.org/wiki/Spherical_coordinate_system

    pos = pos.T

    # r = sqrt(x^2 + y^2 + z^2)
    r = np.sqrt(np.square(pos[0,:]) + np.square(pos[1,:]) +np.square(pos[2,:]))
    # lambda = arc cos (y/z)
    theta = np.arccos(pos[2,:] / r)
    # phi = arctan (z /sqrt(x²+y²))
    phi = np.arctan(pos[1,:] / pos[0,:])

    #print(r.shape)
    #print(theta.shape)
    #print(phi.shape)

    edge_attr_r = np.zeros(len(df_edges))
    edge_attr_theta = np.zeros(len(df_edges))
    edge_attr_phi = np.zeros(len(df_edges))

    for i in range(len(df_edges)):
        #print(int(df_edges.iloc[i]['node1id']))
        #print(int(df_edges.iloc[i]['node2id']))
        coord1_r = r[int(df_edges.iloc[i]['node1id'])]
        coord2_r = r[int(df_edges.iloc[i]['node2id'])]
        coord1_theta = theta[int(df_edges.iloc[i]['node1id'])]
        coord2_theta = theta[int(df_edges.iloc[i]['node2id'])]
        coord1_phi = phi[int(df_edges.iloc[i]['node1id'])]
        coord2_phi = phi[int(df_edges.iloc[i]['node2id'])]

        # calculate edge feature
        edge_attr_r[i] = coord2_r - coord1_r
        edge_attr_theta[i] = coord2_theta - coord1_theta
        edge_attr_phi[i] = coord2_phi - coord1_phi

    df_edges = pd.read_csv(osp.join(folder,id, f'{id}_edges.csv'),sep=';')
    df_edges.insert(1,"edge_r", list(edge_attr_r),True)
    df_edges.insert(2,"edge_theta", list(edge_attr_theta),True)
    df_edges.insert(3,"edge_phi", list(edge_attr_phi),True)

    file_name = osp.join(folder,id, f'{id}_edges_extended.csv')
    df_edges.to_csv(file_name, sep=';')






