# VesselGraph Generation and Statistics

This prepository hosts all source code and utility scripts for the generation of statistical evaluation of the VesselGraph.

## Workflow 

Required input files:

```
BL6J-3mo-4-iso3um_probs_bin0p45.nii.gz (segmentation mask)
BL6J-3mo-4-iso3um_probs_bin0p45_regions.nii.gz (atlas annotation mask)
```

Steps:

1. unzip the NIFTI files
2. run voreen on the segmentation mask
3. post-process voreen results
4. annotate post-processed graph to regions
5. compute statistics

Output:

```
BL6J-3mo-4-iso3um_probs_bin0p45_b_3_0_nodes.csv (raw graph nodes voreen)
BL6J-3mo-4-iso3um_probs_bin0p45_b_3_0_edges.csv (raw graph edges voreen)
BL6J-3mo-4-iso3um_probs_bin0p45_b_3_0_nodes_processed.csv (post-processed graph nodes voreen)
BL6J-3mo-4-iso3um_probs_bin0p45_b_3_0_edges_processed.csv (post-processed graph edges voreen)
BL6J-3mo-4-iso3um_probs_bin0p45_b_3_0_nodes_processed_Atlas.csv (post-processed, Atlas-annotated node list)
BL6J-3mo-4-iso3um_probs_bin0p45_b_3_0_nodes_processed_AtlasGrouped.csv (post-processed, reduced Atlas-annotated node list)
BL6J-3mo-4-iso3um_probs_bin0p45_b_3_0_nodes_processed_AtlasGrouped.csv (post-processed, reduced Atlas-annotated node list)
BL6J-3mo-4-iso3um_probs_bin0p45_b_3_0_nodes_processed_AtlasGrouped.csv (post-processed, reduced Atlas-annotated node list)
BL6J-3mo-4-iso3um_probs_bin0p45_b_3_0_um_closed_loops_edge_len.csv (statistics w.r.t. closed loops and edge length)
```

## Analysis

- Provides methods for computing the number of closed loops.
- Provides methods for computing the average distance to bifurcations.

## Graph Generation 

Provides scripts for running voreen in headless mode. 

## Post Processing

Provides script to merge vessels in voreen-based graph extraction. 

## Visualization

Jupyter Notebooks for visualization processes.

## Visualization Scripts

Provides visualization scripts in order to render graph as vtk.

## Citation

- [arXiv link](https://arxiv.org/abs/2108.13233)
- [Published in NIPS 2021 Dataset & Benchmark Track](https://nips.cc/Conferences/2021/ScheduleMultitrack?event=29873)

```
@misc{paetzold2021brain,
      title={Whole Brain Vessel Graphs: A Dataset and Benchmark for Graph Learning and Neuroscience (VesselGraph)}, 
      author={Johannes C. Paetzold and Julian McGinnis and Suprosanna Shit and Ivan Ezhov and Paul Büschl and Chinmay Prabhakar and Mihail I. Todorov and Anjany Sekuboyina and Georgios Kaissis and Ali Ertürk and Stephan Günnemann and Bjoern H. Menze},
      year={2021},
      eprint={2108.13233},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
Please cite this work if any of our code or datasets are helpful for your research. 