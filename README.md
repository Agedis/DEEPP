DEEPP
===============================================

Code repo for Deep phenotyping of Psychotic Disorders with resting state connectivity 
Created by Javkhlan Byambadorj javkhlan.byambadorj@camh.ca

Project Organization
-----------------------------------

    .
    ├── README.md          <- The top-level 
    ├── .gitignore         <- Files to not upload to github - by default includes /data
    ├── LICENSE            <- usage license if applicable
    ├── data
    │   ├── processed      <- The final dataset (can include subfolders etc)
    │   └── raw            <- The original dataset, generally a link to minimally preprocessed data
    │
    ├── notebooks          <- Jupyter/R notebooks for analysis workflow - Naming should begin with a number, followed by an underscore and a description (e.g. 01_compile_demographics.Rmd)
    │
    ├── docs/references    <- Data dictionaries, manuals, and all other explanatory materials
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment (if applicable)
    │
    ├── code/src           <- Source code for use in this project (virtual environments, bash scripts, etc)

## Data Organization 
The input files used for preprocessing steps in both the **DEEPPI** and **DEEPPD** studies are stored in:
```
/mnt/tigrlab/projects/jbyambadorj/DEEP_study/data/raw
```
In this directory, note that the folder corresponding to the DEEPPI study is a symbolic link that points to:
```
/projects/jbyambadorj/DEEPPI_fmap_intended
```

Data preprocessed using the **fMRIPrep** and **XCP-D** pipelines are stored in:
```
/projects/jbyambadorj/DEEP_study/data/processed
```
In this dir, you will find all the post and preprocessed data along with mriqc csv outputs from the summer. (mriQC was failing for the 
fmap intended for data, so you might need to rerun mriQC by changing the script to appropriate one) 

You will also find parcellated timeseries files (Glasser and Tian atlases) used in the analysis under:
```
/projects/jbyambadorj/DEEP_study/data/processed/inputs
```
Within each study-specific subdirectory, Glasser and Tian parcellated timeseries files from XCP-D are organized as follows:
```
jbyambadorj@noether:/projects/jbyambadorj/DEEP_study/data/processed/inputs/DEEPPI_rerun/sub-CMH00000207$ tree
.
├── ses-01
│   ├── sub_cortical
│   │   ├── sub-CMH00000207_ses-01_task-rest_run-01_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   │   ├── sub-CMH00000207_ses-01_task-rest_run-02_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   │   └── sub-CMH00000207_ses-01_task-rest_run-03_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   └── surface
│       ├── sub-CMH00000207_ses-01_task-rest_run-01_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
│       ├── sub-CMH00000207_ses-01_task-rest_run-02_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
│       └── sub-CMH00000207_ses-01_task-rest_run-03_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
├── ses-02
│   ├── sub_cortical
│   │   ├── sub-CMH00000207_ses-02_task-rest_run-01_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   │   ├── sub-CMH00000207_ses-02_task-rest_run-02_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   │   ├── sub-CMH00000207_ses-02_task-rest_run-03_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   │   └── sub-CMH00000207_ses-02_task-rest_run-04_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   └── surface
│       ├── sub-CMH00000207_ses-02_task-rest_run-01_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
│       ├── sub-CMH00000207_ses-02_task-rest_run-02_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
│       ├── sub-CMH00000207_ses-02_task-rest_run-03_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
│       └── sub-CMH00000207_ses-02_task-rest_run-04_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
├── ses-03
│   ├── sub_cortical
│   │   ├── sub-CMH00000207_ses-03_task-rest_run-01_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   │   ├── sub-CMH00000207_ses-03_task-rest_run-02_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   │   ├── sub-CMH00000207_ses-03_task-rest_run-03_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   │   ├── sub-CMH00000207_ses-03_task-rest_run-04_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   │   └── sub-CMH00000207_ses-03_task-rest_run-05_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   └── surface
│       ├── sub-CMH00000207_ses-03_task-rest_run-01_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
│       ├── sub-CMH00000207_ses-03_task-rest_run-02_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
│       ├── sub-CMH00000207_ses-03_task-rest_run-03_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
│       ├── sub-CMH00000207_ses-03_task-rest_run-04_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
│       └── sub-CMH00000207_ses-03_task-rest_run-05_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
├── ses-04
│   ├── sub_cortical
│   │   ├── sub-CMH00000207_ses-04_task-rest_run-01_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   │   ├── sub-CMH00000207_ses-04_task-rest_run-02_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   │   ├── sub-CMH00000207_ses-04_task-rest_run-03_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   │   ├── sub-CMH00000207_ses-04_task-rest_run-04_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   │   └── sub-CMH00000207_ses-04_task-rest_run-05_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   └── surface
│       ├── sub-CMH00000207_ses-04_task-rest_run-01_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
│       ├── sub-CMH00000207_ses-04_task-rest_run-02_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
│       ├── sub-CMH00000207_ses-04_task-rest_run-03_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
│       ├── sub-CMH00000207_ses-04_task-rest_run-04_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
│       └── sub-CMH00000207_ses-04_task-rest_run-05_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
├── ses-05
│   ├── sub_cortical
│   │   ├── sub-CMH00000207_ses-05_task-rest_run-01_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   │   ├── sub-CMH00000207_ses-05_task-rest_run-02_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   │   ├── sub-CMH00000207_ses-05_task-rest_run-03_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   │   ├── sub-CMH00000207_ses-05_task-rest_run-04_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   │   └── sub-CMH00000207_ses-05_task-rest_run-05_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
│   └── surface
│       ├── sub-CMH00000207_ses-05_task-rest_run-01_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
│       ├── sub-CMH00000207_ses-05_task-rest_run-02_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
│       ├── sub-CMH00000207_ses-05_task-rest_run-03_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
│       ├── sub-CMH00000207_ses-05_task-rest_run-04_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
│       └── sub-CMH00000207_ses-05_task-rest_run-05_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
└── ses-06
    ├── sub_cortical
    │   ├── sub-CMH00000207_ses-06_task-rest_run-01_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
    │   ├── sub-CMH00000207_ses-06_task-rest_run-02_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
    │   └── sub-CMH00000207_ses-06_task-rest_run-03_space-fsLR_seg-Tian_den-91k_stat-mean_timeseries.ptseries.nii
    └── surface
        ├── sub-CMH00000207_ses-06_task-rest_run-01_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
        ├── sub-CMH00000207_ses-06_task-rest_run-02_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
        └── sub-CMH00000207_ses-06_task-rest_run-03_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
```


## Code Repository: 
All analysis scripts (including python scripts for extracting unique vectorized outputs used in the MDS analysis) are stored in:
```
/projects/jbyambadorj/DEEP_study/code
```

The structure is as follows: 
```
jbyambadorj@noether:/projects/jbyambadorj/DEEP_study/code$ tree
.
├── check_ROI.py  - Identifies ROIs with NaN values in pd DataFrame and outputs a summary table
├── DEEPPD_con_matrix.py - Generates RSFC matrices, computes pairwise subject correlations and statistics 
├── DEEPPI_con_matrix.py - Similar to above but for DEEPPI  
├── find_runsAnd_sessions.py - Maps all runs and sessions per subject; outputs a summary table 
├── find_subject_sessions.py - Simplified version; maps sessions per subject only
├── deeppd_mds_noPreS.py - Extract vectorized unique lower triangular values that are used as input for MDS analysis. 
├── deeppi_mds_noPreS.py - Similar to above, but for DEEPPI
├── deeppd_jsonEdit.py - Edits fmap json files to map appropriate fmap run to its corresponding func runs. 
├── deeppi_jsonEdit.py - A script is provided for mapping functional runs to their corresponding fieldmap runs in the DEEPPD study. The mapping follows the structure:
{fmap_run01: [func_run01, func_run02], fmap_run02: [func_run03, func_run04], fmap_run03: [func_run05]}. If a fieldmap run (e.g., fmap_run03) is missing while all five functional runs are present, the script will skip mapping for the unmatched functional run (func_run05). In such cases, fMRIPrep will default to synthetic fieldmap correction for that run instead of using an actual acquired fieldmap.
└── src
    └── activate_myenv.sh - For activating the virtual environment with the necessary packages installed. 
```

The libraries/dependencies that you need for running the code are stored in my virtual env and you can activate the env by typing on terminal
```
source /projects/jbyambadorj/DEEP_study/code/src/activate_myenv.sh
```
In the scripts DEEPPD_con_matrix and DEEPPI_con_matrix, note that R_PreS_ROI is automatically dropped from the pandas DataFrame, regardless of whether it contains NaNs.
To identify which ROIs contain NaNs, use:
```
/projects/jbyambadorj/DEEP_study/code/check_ROI.py
```
This script prompts for a directory path and generates a summary table. For example:

```
Enter the path you want to check for NaN cols
example path: "/projects/jbyambadorj/func_con_matrix/DEEPPI_rerun"
/projects/jbyambadorj/func_con_matrix/DEEPPI_rerun
**************************************************
NA ROIs in the sub-CMH00000001-ses-01 are: ['R_H_ROI', 'L_H_ROI']
NA ROIs in the sub-CMH00000001-ses-02 are: ['R_H_ROI', 'L_H_ROI']
NA ROIs in the sub-CMH00000001-ses-03 are: ['R_H_ROI', 'L_H_ROI']
NA ROIs in the sub-CMH00000001-ses-04 are: ['R_H_ROI', 'L_H_ROI']
NA ROIs in the sub-CMH00000001-ses-05 are: ['R_H_ROI', 'L_H_ROI']
NA ROIs in the sub-CMH00000001-ses-06 are: ['R_H_ROI', 'L_H_ROI']
**************************************************
NA ROIs in the sub-CMH00000069-ses-01 are: ['R_PreS_ROI', 'R_H_ROI', 'L_H_ROI']
NA ROIs in the sub-CMH00000069-ses-02 are: ['R_PreS_ROI', 'R_H_ROI', 'L_H_ROI']
NA ROIs in the sub-CMH00000069-ses-04 are: ['R_PreS_ROI', 'R_H_ROI', 'L_H_ROI']
NA ROIs in the sub-CMH00000069-ses-05 are: ['R_PreS_ROI', 'R_H_ROI', 'L_H_ROI']
NA ROIs in the sub-CMH00000069-ses-06 are: ['R_PreS_ROI', 'R_H_ROI', 'L_H_ROI']
```

Note: The script assumes all functional runs within a session have the same NaN pattern. To speed up processing, it checks only one random run per session, not all runs.




