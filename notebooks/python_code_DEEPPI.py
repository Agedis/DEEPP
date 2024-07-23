from pathlib import Path
import pandas as pd
import nibabel as nib
import seaborn as sns
import numpy as np
import os
import glob
from nilearn.plotting import plot_surf, plot_surf_contours
import matplotlib.pyplot as plt


MY_SUBJECTS = [ 
    "sub-CMH00000001",
    "sub-CMH00000069",
    "sub-CMH00000077",
    "sub-CMH00000079",
    "sub-CMH00000085",
    "sub-CMH00000101"
]   
MY_REGIONS = ['default mode',
 'dorsal attention',
 'frontoparietal',
 'limbic',
 'somatosensory',
 'subcortical',
 'ventral attention',
 'visual']

BASE_DIR = "/projects/jbyambadorj/func_con_matrix/DEEPPI"

class ConnectivityMatrix:
    def __init__(self, session_name: str, subject_name: str) -> None:
        """Initialize the class by providing the session number of the data."""
        self.session_name = session_name
        self.subj_name = subject_name
        self.joined_paths = self._get_ptseries_paths()
        self.atlas_dict = self._make_atlas_df()
        self.nib_map = self._load_nib()
        self.columns_merged = self._get_columns()
        self.ts_sorted = self._get_merged_df()
        self._nan_pd = self.ts_sorted.copy() # initialize this as none 
        self.nan_values = self._extract_NaN()
        self.con_matrix = self._connectivity_matrix()
        self.vector = self.extract_matrix_triangle()
        

    def _get_ptseries_paths(self) -> list:
        """Get the full paths of ptseries files."""
        path_surf = f"{BASE_DIR}/{self.subj_name}/{self.session_name}/surface"
        path_subcortical = f"{BASE_DIR}/{self.subj_name}/{self.session_name}/sub_cortical"

        try:
            path_surf_list = os.listdir(path_surf)
            path_subcortical_list = os.listdir(path_subcortical)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Directory not found: {e}")

        joined = path_subcortical_list + path_surf_list
        joined.sort()

        for i in range(len(joined)):
            if 'Glasser' in joined[i]:
                joined[i] = os.path.join(path_surf, joined[i])
            elif 'Tian' in joined[i]:
                joined[i] = os.path.join(path_subcortical, joined[i])

        return joined

    def _make_atlas_df(self) -> dict:
        """Create atlas dictionary from atlas file."""
        atlas_name = f"{BASE_DIR}/atlas-Glasser_dseg.tsv"
        atlas_df = pd.read_csv(atlas_name, sep='\t')
        sub_atlas_df = atlas_df.loc[:, ["index", "cifti_label", "community_yeo"]]

        atlas_dict = {}
        for i, row in sub_atlas_df.iterrows():
            community = row['community_yeo']
            if community not in atlas_dict:
                atlas_dict[community] = []
            atlas_dict[community].append((row['cifti_label'], row['index']))

        return atlas_dict

    def _load_nib(self) -> dict:
        """Load nibabel objects from ptseries paths."""
        nib_map = {}
        for i in range(0, len(self.joined_paths), 2):
            run_index = i // 2 + 1
            nib_map[f'run_{run_index}'] = [nib.load(self.joined_paths[i]), nib.load(self.joined_paths[i + 1])]
        return nib_map

    def _get_columns(self) -> list:
        """Extract columns from nibabel headers."""
        columns = []
        for key, value in self.nib_map.items():
            columns.append((value[0].header.get_axis(1).name, value[1].header.get_axis(1).name))
        return columns

    def _get_merged_df(self) -> pd.DataFrame:
        """Create and merge DataFrames from nibabel data."""
        ts_data = {}
        for i, (key, value) in enumerate(self.nib_map.items(), start=1):
            ts_data[f'run_{i}'] = [
                pd.DataFrame(value[0].get_fdata(), columns=self.columns_merged[i - 1][0]),
                pd.DataFrame(value[1].get_fdata(), columns=self.columns_merged[i - 1][1])
            ]

        ts_merged = {key: ts[0].join(ts[1]) for key, ts in ts_data.items()}
        ts_mean_merged = pd.concat(ts_merged.values())
        
        # for item in self.atlas_dict:
        #         self.atlas_dict[item] = sorted(self.atlas_dict[item], key=lambda x: x[1])
        for parcels, pos_index in self.atlas_dict.items():
            pos_index.sort(key=lambda x: x[1])

        ts_sorted = ts_mean_merged[['R_V1_ROI']]  

        for key in self.atlas_dict:
            for tup in self.atlas_dict[key]:
                my_roi = tup[0]
                ts_sorted[my_roi] = list(ts_mean_merged[tup[0]]) 

        Tian_ts = ts_mean_merged[self.columns_merged[0][1]]    # columns_merged[2][1] is an array object with names of the parcels in tian atlas 
        ts_sorted = pd.concat([ts_sorted, Tian_ts], axis = 1)

        return ts_sorted 


    def _extract_NaN(self) -> list | None:
        """Count the NaN values in the ts_sorted df and return ROI columns containing NaN vals"""
        NA_count = 0 
        res = [] 
        pd_dataframe = self.ts_sorted 
        for column in pd_dataframe:
            # loop thru rows 
            for value in pd_dataframe[column]:
                if np.isnan(value):  
                    NA_count += 1
                    res.append(column)
                    break # break out from the inner loop so we only count the columns 
        self.ts_sorted = self.ts_sorted.dropna(axis = 1)
        return res 
        
        
    def _connectivity_matrix(self) -> pd.DataFrame:
        """Compute the connectivity matrix."""        
        corZ_pd = np.arctanh(self.ts_sorted.corr())

        for atlas_name in self.atlas_dict:
            for i in range(len(self.atlas_dict[atlas_name])):
                column_name = self.atlas_dict[atlas_name][i][0]
                corZ_pd.rename(columns = {column_name: atlas_name}, inplace = True)

        for column in self.columns_merged[0][1]:
            corZ_pd.rename(columns={column: 'subcortical'}, inplace=True)

        corZ_pd.index = [corZ_pd.columns]
        return corZ_pd

    # for fingerprinting step 
    def extract_matrix_triangle(self) -> list:
        dataframe = self.con_matrix
        n_rows = dataframe.shape[0]
        my_vector = []
        for i in range(n_rows):
            for item in dataframe.iloc[i]:
                if item == np.inf:
                    break
                my_vector.append(item)
        # apply the n(n+1) / 2 formula to check if the length of my_vector matches with all the connectivity values
        if len(my_vector) == ((n_rows -1) * n_rows) / 2:
            return my_vector
        else:
            raise ValueError


    def plot_heatmap(self) -> sns.heatmap:
        """Plot the heatmap of the connectivity matrix."""
        matrix = self._connectivity_matrix()
        heatmap = sns.heatmap(matrix, vmin=-1, vmax=1, cmap="RdBu_r", square=True)
        heatmap.set(title=f'Glasser + Tian {self.subj_name}/{self.session_name}', ylabel='network region')
        return heatmap

    def func_connectivity(self):
        """Run the connectivity function and plot the heatmap."""
        self.plot_heatmap()
        plt.show()



# add a for loop with sessions so that we can
myvar, myvar2 = ConnectivityMatrix("ses-01", "sub-CMH00000101"), ConnectivityMatrix("ses-02", "sub-CMH00000101")

class Region(ConnectivityMatrix):
    def __init__(self, session, subject) -> None:
        super().__init__(session, subject)
        self._index = self._get_roi_index()
        self.mean_values = self.within_mean()
    

    def _get_roi_index(self) -> dict:
        res = {}
        for item in MY_REGIONS:
            region_locations = np.where(self.con_matrix.columns == item)[0] # returns array containing the indices of column_name ROI in conn matrix
            res[item] = min(region_locations) # read np.where if confused
        return res
    

    def within_mean(self) -> dict:
        accumulator = {}
        for region in self._index:
            starting_index = self._index[region]
            n = len(self.con_matrix[region].columns)
            sub_matrix = self.con_matrix[region].iloc[starting_index:starting_index + n]
            value_accumulator = []
            for i in range(n):
                for entry in sub_matrix.iloc[i]:
                    if entry == np.inf:
                        break
                    value_accumulator.append(entry)
            # run the checker below to ensure every entries in the matrix are captured
            # len(value_accumulator) == (n ** 2 - n) / 2
            network_mean = sum(value_accumulator) / len(value_accumulator)
            accumulator[region] = round(network_mean, ndigits = 5)
        return accumulator

    def _roi2network(self):
        """change the ROIs in the ts_sorted df into network names from the atlas."""
        my_ts_copy = self.ts_sorted.copy()
        for key, value in self.atlas_dict.items():
            for i in range(len(value)):
                column_name = value[i][0]
                my_ts_copy.rename(columns = {column_name: key}, inplace = True)
        
        return my_ts_copy

    def between_regions(self):
        ...

class Group:
    def __init__(self, subject_name: str) -> None:
        self.subj_dir = subject_name
        self.sessions = self.get_sessions_list()
        self.ses_map = self.session_to_conn_matrix()
        self._checker = self._test_ses_name()
        self.fingerprint_df = self.subject_fingerprint_df()
        self.fp_corr = self.sub_fingerprint_matrix()
    def get_sessions_list(self) -> list:
        """ Return the list of all sessions in the subject."""
        path_subject = f"{BASE_DIR}/{self.subj_dir}"
        sessions = os.listdir(path_subject)
        for item in sessions:
            if 'ses-0' not in item:
                sessions.remove(item)
        sessions.sort()
        return sessions
    def session_to_conn_matrix(self):
        """ Map each session in the self.sessions to its ConnectivityMatrix object. """
        accumulator = {}
        for ses in self.sessions:
            accumulator[ses] = ConnectivityMatrix(ses, self.subj_dir)
        return accumulator
    def _test_ses_name(self) -> bool:
        for key, value in self.ses_map.items():
            if key != value.session_name:
                return "Ses name in ConMatrix object does not match the ses name in dict key"
        return "No name mismatch"
        
    def subject_fingerprint_df(self) -> pd:
        """ Return a df where rows are sessions of the subject and column is the connectivity
        matrix unique values.
        """
        ses2vector_map = {}
        for key, value in self.ses_map.items():
            ses2vector_map[key] = value.vector
        df = pd.DataFrame(data = ses2vector_map)
        return df
    
    def sub_fingerprint_matrix(self):
        """Create a correlation matrix from the self.fingerprint_df"""
        return self.fingerprint_df.corr()
    

    def make_heatmap_plots(self) -> sns:
        """ Generate func connectivity heatmap plot of each session in the subject."""
        connectivity_objects = list(self.ses_map.values())
        n = len(connectivity_objects)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, 8 * nrows))
        axs = axs.flatten()
        for i in range(n):
            corZ_matrix = connectivity_objects[i].con_matrix
            title = connectivity_objects[i].subj_name + '/' + connectivity_objects[i].session_name
            sns.heatmap(corZ_matrix, vmin=-1, vmax=1, cmap="RdBu_r", ax = axs[i]).set_title(title)
        plt.tight_layout()
        plt.show()

# this part creates different variables using the source sub name 

a = Group("sub-CMH00000101")



global sub 
for item in MY_SUBJECTS:
    # exec(f"{MY_SUBJECTS[i]} = {Group(MY_SUBJECTS[i])}")
    # exec(f"{item} = {Group(item)}")    
    ...

# a = Group("sub-CMH00000101")
# b = Group("sub-CMH00000085")



 

# Should we use all runs in ses for the analysis?? ses-06 has 2 runs whereas others have
# # 5 runs 


# my_hash = {}
# for i in range(len(my_sessions)):
#     my_hash[my_ses[i]] = ConnectivityMatrix(my_sessions[i])

# n = len(my_hash)


# for key, value in my_hash.items():
#     globals()[key] = value.func_connectivity()


# def plot_all(num_session: int): 
#     fig, axs = plt.subplots(nrows = len(my_hash) // 3, ncols = len(my_hash) // 3)
#     fig.set_figwidth(24)
#     sns.heatmap( something ,vmin = -1, vmax = 1, cmap="RdBu_r", ax = axs[0]) 

