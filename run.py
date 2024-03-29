import argparse
import os
import sys
import datetime
import pathlib
from pathlib import Path
from shutil import copyfile
import gzip
import shutil

def gunzip(file):
    print(f'Unzipping {file} to {os.path.splitext(file)[0]}')
    with gzip.open(file, 'rb') as f_in:
        with open(os.path.splitext(file)[0], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

# import custom modules
from post_processing.post_processing import post_processing
from analysis.analyze_loops import analyze_loops
from atlas_annotation.generate_node_atlas_labels import annotate_atlas

parser = argparse.ArgumentParser(description='Control script for running the complete vessel pipeline.')

# general settings and input files
parser.add_argument('-s','--segmentation_mask', help='Path of zipped segmentation mask of the whole brain.', type=str, default = 'segmentation.nii.gz', required=True)
parser.add_argument('-b','--bulge_size', help='Specify bulge size', type=float, default=3.0)
parser.add_argument('-r','--remove_zips', action='store_true', help="whether to remove zips after computation.")

# voreen command line tool options
parser.add_argument('-vp','--voreen_tool_path',help="Specify the path where voreentool is located.", required=True)
parser.add_argument('-wd','--workdir', help='Specify the working directory.', default='.')
parser.add_argument('-td','--tempdir', help='Specify the temporary data directory.', default='.')
parser.add_argument('-cd','--cachedir',help='Specify the cache directory.', default='.')

# atlas annotation
parser.add_argument('-a','--atlas_mask', help='Path of zipped atlas mask of the whole brain.', type=str, default= 'atlas.nii.gz')

# statistics
parser.add_argument('-l','--min_vessel_length', type=float, default=5.0, help='Minimum vessel length')
parser.add_argument('-min', '--min_cycle_length',type=int, help='Specify the minimum required number of nodes.', default=3)
parser.add_argument('-max', '--max_cycle_length', type=int, help='Specify the minimum required number of nodes.', default=15)

args = parser.parse_args()

# check if we are dealing with nii.gz or .nii

smask_path = os.path.abspath(args.segmentation_mask)
amask_path = os.path.abspath(args.atlas_mask)

bulge_size = args.bulge_size
bulge_size_identifier = f'{bulge_size}'
bulge_size_identifier = bulge_size_identifier.replace('.','_')

# directories
workdir = os.path.abspath(args.workdir)
tempdir = os.path.abspath(args.tempdir)
cachedir = os.path.abspath(args.cachedir)

Path(workdir).mkdir(parents=True, exist_ok=True)
Path(tempdir).mkdir(parents=True, exist_ok=True)
Path(cachedir).mkdir(parents=True, exist_ok=True)

# check if file ending is .nii.gz or .nii

if smask_path.endswith('.nii.gz'):
    # graph files
    base_name = os.path.basename(smask_path).replace(".nii.gz", "")
    edge_path = f'{os.path.join(workdir,base_name)}_b_{bulge_size_identifier}_edges.csv'
    node_path = f'{os.path.join(workdir,base_name)}_b_{bulge_size_identifier}_nodes.csv'

else:
    # graph files
    base_name = os.path.basename(smask_path).replace(".nii", "")
    edge_path = f'{os.path.join(workdir,base_name)}_b_{bulge_size_identifier}_edges.csv'
    node_path = f'{os.path.join(workdir,base_name)}_b_{bulge_size_identifier}_nodes.csv'

print(f'Segmentation Mask: {smask_path}')
print(f'Edge List Path: {edge_path}')
print(f'Node List Path: {node_path}')

# graph generation

if os.path.exists(edge_path) and os.path.exists(node_path):
    print("skipping graph generation")

else:

    if os.path.exists(smask_path):
        if smask_path.endswith('nii.gz'):
            if os.path.exists(os.path.splitext(smask_path)[0]):
                print("unzipped version already exists")
            else:
                gunzip(smask_path)
            smask_path = os.path.splitext(smask_path)[0]
        elif smask_path.endswith('.nii'):
            print("unzipped version already exists")
        else:
            sys.exit(f'Segmentation Mask not supported.')
    else:
        sys.exit(f'Segmentation Mask not found.')
    
    bulge_path = f'<Property mapKey="minBulgeSize" name="minBulgeSize" value="{bulge_size}"/>'

    voreen_tool_path = os.path.abspath(args.voreen_tool_path)


    # create temp directory
    temp_directory = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    pathlib.Path(temp_directory).mkdir(parents=True, exist_ok=True)

    workspace_path = 'vesselgraph_workspace.vws'
    copyfile(workspace_path,os.path.join(temp_directory,workspace_path))

    # Read in the file
    with open(os.path.join(temp_directory,workspace_path), 'r') as file :
        filedata = file.read()

    # Replace the target strings
    filedata = filedata.replace("/home/voreen_data/volume.nii", smask_path)
    filedata = filedata.replace("/home/voreen_data/nodes.csv", node_path)
    filedata = filedata.replace("/home/voreen_data/edges.csv", edge_path)
    filedata = filedata.replace('<Property mapKey="minBulgeSize" name="minBulgeSize" value="3" />', bulge_path)

    # Write out workspace file to temp dir
    with open(os.path.join(temp_directory,workspace_path), 'w') as file:
        file.write(filedata)

    workspace_path = os.path.join(os.path.join(os. getcwd(),temp_directory),workspace_path)
    print(workspace_path)

    absolute_temp_path = os.path.join(os.getcwd(),temp_directory)

    # extract graph and delete temp directory
    os.system(f'cd {voreen_tool_path} ; ./voreentool \
    --workspace {workspace_path} \
    -platform minimal --trigger-volumesaves --trigger-geometrysaves  --trigger-imagesaves \
    --workdir {workdir} --tempdir {tempdir} --cachedir {cachedir} \
    ; rm -r {absolute_temp_path}\
    ')


# post processing (vessel merging)
processed_edge_path = edge_path.split('.csv')[0] + "_processed.csv"
processed_node_path = node_path.split('.csv')[0] + "_processed.csv"

if os.path.exists(processed_edge_path) and os.path.exists(processed_node_path):
    print("skipping post processing")
else:
    post_processing(node_path, edge_path, processed_node_path, processed_edge_path)

# only run atlas annotation if atlas mask provided

if os.path.exists(args.atlas_mask):

    path_atlas_nodes = processed_node_path.replace(".csv", "_atlas.csv")
    path_atlas_groups = processed_node_path.replace(".csv", "_atlas_grouped.csv")
    path_atlas_encoded = processed_node_path.replace(".csv", "_atlas_encoded.csv")

    # atlas labelling
    if os.path.exists(path_atlas_nodes) and os.path.exists(path_atlas_groups) and os.path.exists(path_atlas_encoded):
        print("skipping atlas annotation")

    else:

        if os.path.exists(amask_path):
            if amask_path.endswith('nii.gz'):
                if os.path.exists(os.path.splitext(amask_path)[0]):
                    print("unzipped version already exists")
                else:
                    gunzip(amask_path)
                amask_path = os.path.splitext(amask_path)[0]
            elif amask_path.endswith('.nii'):
                print("unzipped version already exists")
            else:
                sys.exit(f'Atlas Mask not supported.')

        print('\nRunning atlas label module')
        onthology_file_path = os.path.join('atlas_annotation', 'AllenMouseCCFv3_ontology_22Feb2021.xml')
        annotate_atlas(onthology_file_path, amask_path, node_path, path_atlas_nodes, path_atlas_groups, path_atlas_encoded)

else:
    print("No atlas provided, skipping atlas annotation.")

# statistics
dataset_name = os.path.commonprefix([processed_node_path, processed_edge_path])
identifier = f"{dataset_name}_stats_vmin_{args.min_vessel_length}_cmin_{args.min_cycle_length}_cmax_{args.max_cycle_length}"
identifier = identifier.replace('__', '_')
identifier = identifier.replace('.', '_') # in float numbers 
identifier = f'{identifier}.csv'

if os.path.exists(identifier):
    print("skipping stats module")
    
else:
    print('\nRunning statistics module')
    analyze_loops(node_path, edge_path, args.min_vessel_length,args.min_cycle_length, args.max_cycle_length, identifier)

# removing unzipped versions (if intended)
if args.remove_zips:
    if os.path.isfile(smask_path):
        os.remove(smask_path)
    if os.path.isfile(amask_path):
        os.remove(amask_path)

