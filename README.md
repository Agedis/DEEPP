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


Input files used for preprocessing steps in both DEEPPI and DEEPPD studies are stored in the directory: 
```
/mnt/tigrlab/projects/jbyambadorj/DEEP_study/data/raw
```
In the above dir, the dir for DEEPPI study is left as a symbolic link pointing to:
```
/projects/jbyambadorj/DEEPPI_fmap_intended
```

The preprocessed data from fMRIprep and xcp-D pipelines are located in the directory: 
```
/projects/jbyambadorj/DEEP_study/data/processed
```
In this dir, you will find all the post and preprocessed data along with mriqc csv outputs from the summer. (mriQC was failing for the 
fmap intended for data, so you might need to rerun mriQC by changing the script to appropriate one) 
Furthermore in this dir, you will find Glasser and Tian only parcellated timeseries files that I used mainly for the analysis: 
```
/projects/jbyambadorj/DEEP_study/data/processed/inputs
```
Inside this dir, if you go to one of the study specific subdir, you will find the Glasser and Tian parcellated timeseries xcp-D outputs organized in the following way: 
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





All the code used in analysis up until MDS analysis are stored in 
```
/projects/jbyambadorj/DEEP_study/code/src
```
The libraries that you need for running the code are stored in my virtual env and you can activate the env by typing on terminal
```
source /projects/jbyambadorj/DEEP_study/code/src/activate_myenv.sh


```
Python scripts DEEPPD_con_matrix.py and DEEPPI_con_matrix.py are for creating the Pairwise Distance Matrices and calculation of statistical values such as within-subject mean and between-subjects mean.


Note that the pd Dataframe objects that are created from the parcellated timeseries file drops the PreS ROI regardless of whether this ROI is NaN or not in the study. 
You can check which ROIs are NaNs only only by running the script: 
```
/projects/jbyambadorj/DEEP_study/code/check_ROI.py
```
It will generate a table of which ROIs have NaNs only as shown in the small example below: 

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

The script assumes that all functional runs within a session will likely have same NaNs to reduce the search time. So keep in mind that the script is not thoroughly checking every single run in the subject data and instead just chooses 
1 random func data in a given session. 





