# from https://deepnote.com/@deepnote/3D-network-visualisations-using-plotly-oYxeN6UXSye_3h_ulKV2Dw

#Import the required packages
import networkx as nx 
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
import argparse
import os


def get_values_as_tuple(dict_list, keys):
        return [tuple(d[k] for k in keys) for d in dict_list]   

parser = argparse.ArgumentParser(description='visualizes voreen csv based graph in plotly as HTML visualization.')
parser.add_argument('-n','--node_list', help='Name of the node list csv file.', required=True)
parser.add_argument('-e','--edge_list', help='Filtering condition: average radius of the vessel.', required=True)
parser.add_argument('-o','--output_directory', help='Output name of converted vtk files.', required=True)

# read command line arguments
args = vars(parser.parse_args())
node_path = args['node_list']
edge_path = args['edge_list']
output_path = os.path.abspath(args['output_directory'])

# read Voreen graph and convert to nx format: https://networkx.org/documentation/stable/tutorial.html

df_nodes = pd.read_csv(node_path,sep=';', usecols=['pos_x', 'pos_y', 'pos_z']).abs() # only absolute values
#node_attr = tuple(df_nodes.to_dict("index").items()) 
node_attr = df_nodes.to_dict("index").items()

df_edges = pd.read_csv(edge_path,sep=';')
edge_items = df_edges[['length','distance','curveness','maxRadiusAvg']].to_dict("index").items()
edge_attr = [tuple([int(df_edges.iloc[item[0]]['node1id']),int(df_edges.iloc[item[0]]['node2id']),item[1]]) for item in edge_items]

#print(node_attr)
#print(edge_attr)

# creating networkx graph

voreen_graph = nx.Graph()
voreen_graph.add_nodes_from(node_attr)
voreen_graph.add_edges_from(edge_attr)

#print(list(nx.connected_components(voreen_graph)))

num_nodes = len(df_nodes.index)
num_edges = len(df_edges.index)

node_pos = list(pd.read_csv(node_path,sep=';', usecols=['pos_x', 'pos_y', 'pos_z']).abs().to_numpy()) # only absolute values
node_pos_dict = dict([(i, node_pos[i]) for i in range(num_nodes)])

# Voreen
x_nodes = list(pd.read_csv(node_path,sep=';', usecols=['pos_x']).abs().to_numpy().flatten())
y_nodes = list(pd.read_csv(node_path,sep=';', usecols=['pos_y']).abs().to_numpy().flatten())
z_nodes = list(pd.read_csv(node_path,sep=';', usecols=['pos_z']).abs().to_numpy().flatten())


df_edges = pd.read_csv(edge_path,sep=';')
edge_index = np.array(df_edges[['node1id','node2id']].to_numpy())

edges = []
x_edges=[]
y_edges=[]
z_edges=[]

df_nodes = pd.read_csv(node_path,sep=';', usecols=['pos_x', 'pos_y', 'pos_z']).abs() # only absolute values

pos = np.array(df_nodes[['pos_x', 'pos_y', 'pos_z']].to_numpy())
pos = np.abs(pos)

for i in range (0, len(edge_index)):
    edge_1 = edge_index[i,0]
    edge_2 = edge_index[i,1]
    coord_1 = pos[edge_1]
    coord_2 = pos[edge_2]
    x_edges.append([coord_1[0],coord_2[0],None])
    y_edges.append([coord_1[1],coord_2[1],None])
    z_edges.append([coord_1[2],coord_2[2],None])

x = [item for sublist in x_edges for item in sublist]
y = [item for sublist in y_edges for item in sublist]
z = [item for sublist in z_edges for item in sublist]

#we  need to create lists that contain the starting and ending coordinates of each edge.
x_edges=x
y_edges=y
z_edges=z

# take a random edge

index = np.random.randint(num_edges, size=1)

# keep trace of the two nodes we are examining

node_A = int(df_edges.iloc[index]['node1id']) # determine the index
node_B = int(df_edges.iloc[index]['node2id']) # determine the real index

print(node_A)
print(node_B)


node_info = pd.read_csv(node_path,sep=';',usecols=['id','pos_x','pos_y','pos_z']).to_dict("records")
node_text = [f'ID: {item["id"]}, X:{item["pos_x"]}, Y:{item["pos_y"]}, Z:{item["pos_z"]}' for item in node_info]

edge_info = pd.read_csv(edge_path,sep=';',usecols=['length','distance','curveness','avgRadiusAvg']).to_dict("records")
edge_text = [f'Length: {item["length"]}, Dist:{item["distance"]}, Curveness:{item["curveness"]}, Rad.:{item["avgRadiusAvg"]}' for item in edge_info]

#import plotly.express as px
#df = px.data.iris() # iris is a pandas DataFrame
#print 


#create a trace for the edges
trace_edges = go.Scatter3d(x=x_edges,
                        y=y_edges,
                        z=z_edges,
                        mode='lines',
                        line=dict(color='black', width=1), 
                        #line = dictionary,
                        text = edge_text,
                        #hoverinfo='text')
                        hoverinfo='text')



#create a trace for the nodes
trace_nodes = go.Scatter3d(x=x_nodes,
                        y=y_nodes,
                        z=z_nodes,
                        mode='markers',
                        marker=dict(symbol='circle',
                                    size=1,
                                    #color=community_label, #color the nodes according to their community
                                    #colorscale=['lightgreen','magenta'], #either green or mageneta
                                    line=dict(color='black', width=0.5)),
                        text=node_text,
                        hoverinfo='text')

#create a trace for the nodes
'''
trace_nodes = go.Scatter3d(nodes, x='pos_x',
                        y='pos_y',
                        z='pos_z',
                        mode='markers',
                        marker=dict(symbol='circle',
                                    size=1,
                                    #color=community_label, #color the nodes according to their community
                                    #colorscale=['lightgreen','magenta'], #either green or mageneta
                                    line=dict(color='black', width=0.5)),
                        #text=node_text,
                        hoverinfo='none')

'''


trace_A = go.Scatter3d(x=[x_nodes[node_A]],
                    y=[y_nodes[node_A]],
                    z=[z_nodes[node_A]],
                    mode='markers',
                    name='node_A',
                    marker=dict(symbol='circle',
                                size=3,
                                color='darkblue',
                                line=dict(color='black', width=0.5)
                                ),
                    text = ['node_A'],
                    hoverinfo = 'text')

trace_B = go.Scatter3d(x=[x_nodes[node_B]],
                        y=[y_nodes[node_B]],
                        z=[z_nodes[node_B]],
                        mode='markers',
                        name='node_B',
                        marker=dict(symbol='circle',
                                    size=3,
                                    color='darkgreen',
                                    line=dict(color='black', width=0.5)
                                    ),
                        text = ['nodeB'],
                        hoverinfo = 'text')


#we need to set the axis for the plot 
axis = dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title='')

#also need to create the layout for our plot
layout = go.Layout(title="Region of Interest",
                width=650,
                height=625,
                showlegend=False,
                scene=dict(xaxis=dict(axis),
                        yaxis=dict(axis),
                        zaxis=dict(axis),
                        ),
                margin=dict(t=100),
                hovermode='closest')

#Include the traces we want to plot and create a figure
data = [trace_edges, trace_nodes,trace_A, trace_B]
fig = go.Figure(data=data, layout=layout)

# https://stackoverflow.com/questions/59868987/plotly-saving-multiple-plots-into-a-single-html

#with open('test.html', 'a') as f:
#    f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
#    f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
#    f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

pio.write_html(fig,'test.html')
#fig.show()
os.system("firefox test.html")

