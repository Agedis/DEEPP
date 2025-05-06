from pathlib import Path
import pandas as pd
import nibabel as nib
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt

BASE_DIR = "/projects/jbyambadorj/func_con_matrix/DEEPPI_rerun"


MY_SUBJECTS = []   

# populate the mysubjects list 

def get_subjects(mydir: str) -> None:
    """ Populate the MY_SUBJECTS list by existing subject IDs in mydir (name of your directory with subject data). 
    Precondition: mypath needs to be a valid input that can be transformed into path object. 
    """
    mypath = os.path.join(mydir)
    lst = os.listdir(mypath)
    for item in lst:
        if "sub-CMH" in item:
            MY_SUBJECTS.append(item)

get_subjects("/projects/jbyambadorj/func_con_matrix/DEEPPI_rerun")
MY_SUBJECTS.sort()

# Remove sub207 since this subject has many blank/NaN ROIs than others 
MY_SUBJECTS.remove('sub-CMH00000207')


MY_REGIONS = ['default mode',
 'dorsal attention',
 'frontoparietal',
 'limbic',
 'somatosensory',
 'subcortical',
 'ventral attention',
 'visual']

MY_ID = ['PI01', 'PI69', 'PI77', 'PI79', 'PI85', 'PI101'] # contains the id of every subject in DEEPPI 
                                             # this list is a modified version of the MY_SUBJECTS


class ConnectivityMatrix:
    def __init__(self, session_name: str, subject_name: str) -> None:
        """Initialize the class by providing the session number of the data."""
        self.session_name = session_name
        self.subj_name = subject_name
        self.joined_paths = self._get_ptseries_paths()
        self.atlas_dict = self._make_atlas_df()
        self.nib_map = self._load_nib()
        self.columns_merged = self._get_columns()
        self.ts_of_runs = self._get_runs()
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
        """Create atlas dictionary from atlas file. Atlas dictionary
        keys are network regions (e.g. visual) and values are ROI and their indices sorted in ascending order.
        Output network to ROI map is in the form:
        {'visual': [('R_V1_ROI', 1), ('R_V6_ROI', 3), ...],
        'somatosensory': [('R_4_ROI', 8), ('R_3b_ROI', 9), ...],
        'default mode': [('R_RSC_ROI', 14), ('R_POS2_ROI', 15), ('R_SFL_ROI', 26), ...]
        ...}
        """
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
            break 
	    # break the loop with one iteration since cols in nibabel is invariant same across all ptseries files.
        return columns

    def _get_runs(self) -> dict[pd.DataFrame]:
        """ Create pd df from the nib map objects mapped onto runs."""
        ts_data = {}
        for i, (key, value) in enumerate(self.nib_map.items(), start=1):
            ts_data[f'run_{i}'] = [
                pd.DataFrame(value[0].get_fdata(), columns=self.columns_merged[0][0]),
                pd.DataFrame(value[1].get_fdata(), columns=self.columns_merged[0][1])
            ]

        ts_merged = {key: ts[0].join(ts[1]) for key, ts in ts_data.items()}
        return ts_merged


    def _get_merged_df(self) -> pd.DataFrame:
        """Create and merge DataFrames from nibabel data.
        To sort the columns/ROIs in merged timeseries across runs, this method adds one ROI/col from original df at a time
        based on columns sorted by regions in self.atlas_dict."""

        # Merge time series from runs
        ts_mean_merged = pd.concat(self.ts_of_runs.values(), ignore_index=True)

        # Ensure atlas_dict's indices are sorted
        for parcels, pos_index in self.atlas_dict.items():
            pos_index.sort(key=lambda x: x[1])

        # Start with the base column, making an explicit copy to avoid SettingWithCopyWarning
        frames = []

        # Iterate over atlas_dict and collect DataFrames for each ROI
        for key in self.atlas_dict:
            for tup in self.atlas_dict[key]:
                my_roi = tup[0]
                # Extract the ROI column as a DataFrame and add to list
                frames.append(ts_mean_merged[[my_roi]].copy())

        # Get the Tian atlas columns from original dataframe
        Tian_ts = ts_mean_merged[self.columns_merged[0][1]]
        frames.append(Tian_ts)
        ts_sorted = pd.concat(frames, axis = 1)
        return ts_sorted


    def _extract_NaN(self) -> list[str]:
        """Mutate the sorted timeseries DataFrame by droppin NaN values 
        and return a list of ROI columns containing those NaN vals."""
        
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
        corZ_pd = self.ts_sorted.corr()

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
                if item >= 1:
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


class Region(ConnectivityMatrix):
    """
    Extends the Connectivity Matrix super class. 
    """
    def __init__(self, session, subject) -> None:
        super().__init__(session, subject)
        self._index = self._get_roi_index()
        self.mean_values = self.within_mean()
        self.network_df = self._roi2network()
    

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
        
        for column_name in my_ts_copy:
            if column_name not in MY_REGIONS:
                my_ts_copy.rename(columns = {column_name: 'subcortical'}, inplace = True)
            
            

        return my_ts_copy

    def between_regions(self):
        """Create a df where each row contains the region mean of that row."""
        n_rows = self._roi2network().shape[0]
        res = {}

        for region in MY_REGIONS:
            row_mean_accumulator = [] 
            region_df = self.network_df[region]
            for i in range(n_rows):
                row_mean = region_df.iloc[i].mean()
                row_mean_accumulator.append(row_mean)
            res[region] = row_mean_accumulator
        
        return pd.DataFrame(data = res)
    
    def between_network_corr(self):
        return self.between_regions().corr()
    
    def between_network_heatmap(self):
        heatmap = sns.heatmap(self.between_network_corr(), cmap="RdBu_r", square=True, vmax = 1, vmin = -1)
        plt.show()


# myvar, myvar2 = Region("ses-01", "sub-CMH00000101"), ConnectivityMatrix("ses-02", "sub-CMH00000101")


def truncate_sub_id(subject_id: str) -> str:
    """ Truncate a give input name into its shorter form with the corresponding 
    study.
    Precondition: 
        - input subject_id is of the form sub-CMH00000001 
        - every subject_id contains the header sub-CMH
    """
    # skip first 7 chars 
    unique_id = subject_id[7:]
    char = unique_id[0]
    n = len(unique_id)
    count = 0 
    for char in unique_id:
        if char == '0':
            count += 1 
        else:
            break # break when first non_zero char is spotted 
    id_without_zeros = unique_id[count:]
    # special case if subject id without trailing zeros is just 1 digit. 
    if len(id_without_zeros) == 1:
        # add 0 before 
        return "sub" + "0" + id_without_zeros
  
    return "sub" + id_without_zeros


class SubjectGroup:
    def __init__(self, subject_name: str) -> None:
        self.subj_dir = subject_name
        self.sessions = self.get_sessions_list()
        self.ses_map = self.session_to_conn_matrix()
        self._checker = self._test_ses_name()
        self.fingerprint_df = self.subject_fingerprint_df()
        self.fp_corr = self.sub_fingerprint_matrix()
        self.fp_plot = self.fp_heatmap()


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
            sns.heatmap(np.tanh(corZ_matrix), vmin=-1, vmax=1, cmap="RdBu_r", ax = axs[i]).set_title(title)
        plt.tight_layout()
        plt.show()

    """"
    **************************************************************************************
    Below part is for constructing fingerprint matrices from the subject data 
    **************************************************************************************
    """

    def _sub_and_ses_column_names(self) -> None:
        """ Change the column names in the self.fingerprint_df by adding subject name before '/'
        Example:
            if column names were ses-01; ses-02; etc -> sub101/ses-01; sub101/ses-02; etc 
        
        """
        # use the helper defined before class for changing subject ID to its truncated ver
        sub_name = truncate_sub_id(self.subj_dir)

        column_names_vector = [] 
        for item in self.sessions:
            new_column = sub_name + '/' + item
            column_names_vector.append(new_column)
        return column_names_vector
        # the col names of the df are not changed in this function. Instead it happens in the next method. 
        # when df.columns is reassigned.         

    def subject_fingerprint_df(self) -> pd:
        """ Return a df where rows are sessions of the subject and column is the connectivity
        matrix unique values.
        """
        ses2vector_map = {}
        for key, value in self.ses_map.items():
            ses2vector_map[key] = value.vector
        df = pd.DataFrame(data = ses2vector_map)

        df.columns = self._sub_and_ses_column_names() # this changes the columns
        return df


    def sub_fingerprint_matrix(self):
        """Create a correlation matrix from the self.fingerprint_df"""
        return self.fingerprint_df.corr()
    
    def fp_heatmap(self):
        """Construct the heatmap plot form subject fingerprint matrix."""
        heatmap = sns.heatmap(self.fp_corr, vmin=-1, vmax=1, cmap="RdBu_r", square=True)

        heatmap.set(title=f'{self.subj_dir} fingerprint', ylabel='sessions')



# a = SubjectGroup("sub-CMH00000101")



subjects_dict = {}
for item in MY_SUBJECTS:

    key_name = truncate_sub_id(item)
    subjects_dict[key_name] = SubjectGroup(item)


class AcrossSubjects(SubjectGroup):
    """
    This class is for doing group level analysis using every subjects in the data.
    
    """

    def __init__(self, subjects: dict) -> None:
        self.subjects = subjects
        self.joined_dfs = self.join_fingerprint_dfs()
        self.matrix = self.corZ_matrix()
        self.subjects_mean = self.mean_and_stdev_within_subjects()
        self.labels_renamed_matrix = self.df_change_labels()

    def join_fingerprint_dfs(self) -> pd:
        """ Join all the fingerprint dfs in the subjects_dict."""
        accumulator = [] 
        for key, value in self.subjects.items():
            accumulator.append(value.fingerprint_df)

        joined_df = accumulator[0]
        n = len(accumulator)
        for i in range(1, n):
            joined_df = joined_df.join(accumulator[i], how = 'outer')
        
        return joined_df
    
    def corZ_matrix(self) -> pd:
        """ Construct a correlation matrix from the self.joined_dfs dataframe. """
        return self.joined_dfs.corr(numeric_only=float)
    
    def _truncate_labels(self) -> list[str]:
        """ Shorten the labels in a pd df by getting rid of ses identifiers.
        Precondition:
            Every label in the list contains "/"
            Input is the self.matrix
        Postcondition:
            Return label names truncated in the same order as input matrix.
        """
        res = []
        labels_list = self.matrix.index.to_list()
        for item in labels_list:
            # split the label at "/" (since IDs are of the form PD12/ses03)
            subject_id_without_ses = item.split("/")[0]
            res.append(subject_id_without_ses)
        return res
    
    
    def df_change_labels(self) -> pd.DataFrame:
        """ Precondition: len of new_labels is same as len of pd.DataFrame.labels and
        arranged in the same order. """
        temp = self.matrix.copy()
        old_labels = temp.index.to_list()
        n = len(old_labels)
        new_labels = self._truncate_labels()
        mymap = {}
        for i in range(n):
            old_label = old_labels[i]
            new_label = new_labels[i]
            mymap[old_label] = new_label
        # do renaming for both x and y axis labels
        temp.rename(mymap, axis = "index", inplace = True)
        temp.rename(mymap, axis = "columns", inplace = True)
        return temp

    def heatmap(self) -> sns:
        # when constructing the heatmap np.arctanh function is called to fisher untransform
        # the values and use the Pearson corr values
        heatmap = sns.heatmap(self.matrix, cmap = "RdBu_r", vmax = 1.1, vmin = -1.1)
        heatmap.set_title("DEEPPD Pairwise Correlation")
        plt.savefig(BASE_DIR + "/heatmap.png")


    """
    ***************************************************************************
    Below methods are for running statistical analysis of the pairwise correlation matrix.
    ***************************************************************************
    """
    def _extract_vector_mean(self, corr_matrix_argument: pd) -> pd:
        """ Extract the vector from subject specific corr_matrix. This is used as
        a helper for the method described below for finding the mean value
        and standard deviation in the subject.

        """
        res = []
        n = corr_matrix_argument.shape[0]
        for i in range(n): # loop by row index in the corr_marix df
            for value in corr_matrix_argument.iloc[i]: # loop through values in the row
                if value >= 1:
                    break
                res.append(value)
        mean, stdev = np.mean(res), np.std(res)
        return float(mean), float(stdev)


    def mean_and_stdev_within_subjects(self) -> dict[str: list[float]]:
        """
        Calculate the mean and standard deviation between sessions within subject.
        Returns:
        {'sub01': (sessions_mean, sessions_stdev),
         'sub02': (sessions_mean, sessions_stdev),
          ...
        }
        """
        res = {}
        for key, value in subjects_dict.items():
            mean_and_stdev= self._extract_vector_mean(value.fp_corr)
            res[key] = mean_and_stdev
        return  res

    def average_stats_across_subjects(self) -> str:
        """
        From the above subject specific within mean and std (calculated for each subject)
        find the average of those values i.e. find avg of the means and avg of stdevs

        Returns:
            float: The standard deviation of the means from all subjects.
            str: "mean: {float}, stdev: {float}"
        """
        res = []
        for key, val in self.subjects_mean.items():
            mean_value = val[0]
            res.append(mean_value) # since mean is stored as the first object of the tuple in dict value.
        stdev = np.std(res)
        average = np.mean(res)
        return f"mean: {float(average)},stdev: {float(stdev)}"

temp = AcrossSubjects(subjects_dict)


# functions for doing between subjects functional connectivity mean and stdev analysis 


def extract_between_subject_values(pairwise_matrix: pd.DataFrame) -> dict[str: list[float]]:
    """ Extract unique values that are non-self in the pairwise_matrix and return
     a dictionary mapping those values to their corresponding subject identifier.
    Precondition:
        Labels in pairwise_matrix  need to be repeating identifiers for
        distinguishing self vs non-self.
        Input is AcrossSubjects.labels_renamed_matrix
    Example:
        {"sub01": [0.490248, 0.75186, ...] ,
         "sub02": [0.843243, 0.64231, ...],
         ...}
    """
    res = {}
    mylabels = pairwise_matrix.index.to_list()
    n = len(mylabels)
    unique_ids = []
    for i in range(n):
        if mylabels[i] not in unique_ids:
            unique_ids.append(mylabels[i])

    for item in unique_ids:
        subject_cols = pairwise_matrix[item]
        # drop the subject ids from rows in the column sub data
        between_subject_df = subject_cols.drop(item, axis = 0)
        # map subj to between subjects mean
        res[item] = [np.mean(between_subject_df)]
        # map subj to between subjects stdev
        values = between_subject_df.to_numpy().flatten()
        res[item].append(np.std(values))
    return res


def mean_and_stdev_between_subjects(subject2mean_and_stdev_map: dict[str: list[float]]) -> list[float]:
    """Return mean of the averages and stdevs calculated from between subject comparison"""
    avg_means = []
    avg_stdevs = []
    for val in subject2mean_and_stdev_map.values():
        avg_means.append(val[0])
        avg_stdevs.append(val[1])
    return (np.mean(avg_means),  np.mean(avg_stdevs))
