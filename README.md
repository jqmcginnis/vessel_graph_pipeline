# VesselGraph Pipeline

This prepository hosts source code for an automated voreen-based pipeline covering graph generation, post processing, atlas annotation and statistical analysis. 

## Workflow 

Install voreen to your local linux machine, following the [installation instructions]: https://github.com/jqmcginnis/vessel_graph_pipeline/tree/main/installation

Required input files:

```
BL6J-3mo-4-iso3um_probs_bin0p45.nii.gz (segmentation mask)
```
Optional:

```
BL6J-3mo-4-iso3um_probs_bin0p45_regions.nii.gz (atlas annotation mask)
```

Pipeline Steps:

1. unzip NIFTI files
2. run voreen on the segmentation mask
3. post-process and merge voreen results 
4. annotate post-processed graph to alan mouse brain atlas
5. compute statistics
6. remove unzipped NIFTI files

Run Pipeline:

```
python3 run.py -vp /home/home/johannes_julian/voreen/binaries/voreen-src-unix-nightly/bin --segmentation_mask BL6J-12mo-10_iso3um_probs_bin0p49.nii.gz --atlas_mask BL6J-12mo-10_iso3um_probs_bin0p49_regions.nii.gz
```

Output:

```
BL6J-3mo-4-iso3um_probs_bin0p45_b_3_0_nodes.csv (raw graph nodes voreen)
BL6J-3mo-4-iso3um_probs_bin0p45_b_3_0_edges.csv (raw graph edges voreen)
BL6J-3mo-4-iso3um_probs_bin0p45_b_3_0_nodes_processed.csv (post-processed graph nodes voreen)
BL6J-3mo-4-iso3um_probs_bin0p45_b_3_0_edges_processed.csv (post-processed graph edges voreen)
BL6J-3mo-4-iso3um_probs_bin0p45_b_3_0_nodes_processed_atlas.csv (post-processed, Atlas-annotated node list)
BL6J-3mo-4-iso3um_probs_bin0p45_b_3_0_nodes_processed_atlas_grouped.csv (post-processed, reduced Atlas-annotated node list)
BL6J-3mo-4-iso3um_probs_bin0p45_b_3_0_nodes_processed_atlas_encoded.csv (one-hot encoded region file)
BL6J-3mo-4-iso3um_probs_bin0p45_b_3_0_stats_vmin_5_0_cmin_3_cmax_15.csv(statistics w.r.t. closed loops and edge length)
```

## Analysis

- Provides methods for computing the number of closed loops.

## Atlas Annotation

- Annotation to Alan Mouse Brain Atlas.

## Post Processing

- Provides script to merge vessels in voreen-based graph extraction. 
- The exact method is described in greater detail in our paper [arXiv link](https://arxiv.org/abs/2108.13233).

## Visualization

- Jupyter Notebooks for visualization processes.

## Visualization Scripts

- Provides visualization scripts in order to render graph as vtk graphs.

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
