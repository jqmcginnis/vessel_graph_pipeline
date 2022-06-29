import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Filter microvessels')
parser.add_argument('-e','--edge_list', help='Edge List to be processed.', required=True)
parser.add_argument('-n','--node_list', help='Node List to be processed.', required=True)
parser.add_argument('-d','--display_fit',help='Whether to display the fit or not.',action='store_true')
parser.add_argument('-r','--radius', help='Threshold radius for microvessels', required=False, default=5.0,type=float)
args = parser.parse_args()

# read unfiltered edge list
df_edges = pd.read_csv(args.edge_list,sep=';')
df_edges = df_edges.sort_values(by="avgRadiusAvg",ascending=True)

# filter all edges below 15um
# df_edges = df_edges.loc[df_edges['avgRadiusAvg'] < args.radius]

edge_path = args.edge_list.split('.csv')[0] + "_avg_distance.csv"
node_path = args.node_list.split('.csv')[0] + "_avg_distance_microvessels.csv"

# obtain list of all nodes included in the microvessel graph
node_ids = np.array((np.array(df_edges[['node1id']]),np.array(df_edges[['node2id']]))).flatten()
# get unique nodes and sort in ascending order
node_ids = np.sort(np.unique(node_ids), axis=0)

# read unfiltered edge list
df_nodes = pd.read_csv(args.node_list,sep=';')
df_nodes = df_nodes.iloc[node_ids]

# compute average distance to other bifurcations
# for each node extract all edges

node_list = list(df_nodes["id"])

num_of_bifurcations = []
mean_to_bif =[]

for node in node_list:
	# get all relevant edges
    edges = df_edges[(df_edges['node1id'] == node) | (df_edges['node2id'] == node) ]
    num_of_bifurcations.append(len(edges.index))
    mean_to_bif.append( edges["distance"].mean())

df_nodes["avg_distance_to_bifurcations"] = mean_to_bif
df_nodes["num_bifurcations_to_microvessels"] = num_of_bifurcations

# save new csvs
# df_edges.to_csv(edge_path,index=False,sep=';')
df_nodes.to_csv(node_path,index=False,sep=';')
