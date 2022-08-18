import argparse
import os
import datetime
import pathlib
from shutil import copyfile

parser = argparse.ArgumentParser(description='Control script for running the complete vessel pipeline.')

# general settings and input files
parser.add_argument('-s','--segmentation_mask', help='Path of zipped segmentation mask of the whole brain.', required=True)
parser.add_argument('-a','--atlas_mask', help='Path of zipped atlas mask of the whole brain.', default="", type=str)
parser.add_argument('-b','--bulge_size', help='Specify bulge size', type=float, required=True)

# voreen command line tool options
parser.add_argument('-vp','--voreen_tool_path',help="Specify the path where voreentool is located.", required=True)
parser.add_argument('-wd','--workdir', help='Specify the working directory.', required=True)
parser.add_argument('-td','--tempdir', help='Specify the temporary data directory.', required=True)
parser.add_argument('-cd','--cachedir',help='Specify the cache directory.', required=True)

args = parser.parse_args()

input_image_path = args.segmentation_mask
bulge_size = args.bulge_size

workdir = args.workdir
tempdir = args.tempdir
cachedir = args.cachedir

voreen_tool_path = args.voreen_tool_path
workspace_path = 'vesselgraph_workspace.vws'

volume_path = input_image_path
bulge_size_identifier = f'{bulge_size}'
bulge_size_identifier = bulge_size_identifier.replace('.','_')
edge_path = f'{os.path.join(workdir,os.path.splitext(input_image_path)[0])}_b_{bulge_size_identifier}_edges.csv'
node_path = f'{os.path.join(workdir,os.path.splitext(input_image_path)[0])}_b_{bulge_size_identifier}_nodes.csv'
graph_path = f'{os.path.join(workdir,os.path.splitext(input_image_path)[0])}_b_{bulge_size_identifier}_graph.vvg.gz'

print(f'{volume_path}')
print(f'{edge_path}')
print(f'{node_path}')
print(f'{graph_path}')

bulge_path = f'<Property mapKey="minBulgeSize" name="minBulgeSize" value="{bulge_size}"/>'

# create temp directory

temp_directory = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
pathlib.Path(temp_directory).mkdir(parents=True, exist_ok=True)

voreen_workspace = 'vesselgraph_workspace.vws'
copyfile(workspace_path,os.path.join(temp_directory,voreen_workspace))

# Read in the file
with open(os.path.join(temp_directory,voreen_workspace), 'r') as file :
    filedata = file.read()

# Replace the target string
filedata = filedata.replace("/home/voreen_data/volume.nii", volume_path)
filedata = filedata.replace("/home/voreen_data/nodes.csv", node_path)
filedata = filedata.replace("/home/voreen_data/edges.csv", edge_path)
filedata = filedata.replace('<Property mapKey="minBulgeSize" name="minBulgeSize" value="3" />', bulge_path)

# Write the file out again
with open(os.path.join(temp_directory,voreen_workspace), 'w') as file:
    file.write(filedata)

workspace_path = os.path.join(os.path.join(os. getcwd(),temp_directory),voreen_workspace)
print(workspace_path)

absolute_temp_path = os.path.join(os.getcwd(),temp_directory)

# extract graph and delete temp directory

os.system(f'cd {voreen_tool_path} ; ./voreentool \
--workspace {workspace_path} \
-platform minimal --trigger-volumesaves --trigger-geometrysaves  --trigger-imagesaves \
--workdir {workdir} --tempdir {tempdir} --cachedir {cachedir} \
; rm -r {absolute_temp_path}\
')


