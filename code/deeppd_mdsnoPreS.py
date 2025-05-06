import pandas as pd
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = "/projects/jbyambadorj/func_con_matrix/DEEPPD_rerun"

class ConnectivityMatrix:
    """Class for creating resting state functional connectivity (RSFC) matrices from parcellated timeseries outputs from xcp-D. 
    """

    def __init__(self, session_name: str, subject_name: str) -> None:
        """
        Parameters:
            session_name: ses-0$i (where 1 <= i <= 4)
            subject_name: full ID of the subject e.g. sub-CMH00000010
        """
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

    def _make_atlas_df(self) -> dict[str, (str, int)]:
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
        based on columns sorted by regions in self.atlas_dict.
        """
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
        """Mutate the sorted timeseries DataFrame by droppin NaN values and return a list of ROI columns containing those NaN vals."""
        res = []
        pd_dataframe = self.ts_sorted
        for column in pd_dataframe:
            # loop thru rows
            for value in pd_dataframe[column]:
                if np.isnan(value):
                    res.append(column)
                    break # break out from the inner loop so we only count the columns
        self.ts_sorted = self.ts_sorted.drop("R_PreS_ROI", axis=1, errors="ignore")
        return res


    def _connectivity_matrix(self) -> pd.DataFrame:
        """Compute the connectivity matrix by calculating Fischer Z value correlations."""
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


var = ConnectivityMatrix("ses-02", "sub-CMH00000010")


def find_sessions(mydirectory_name: str) -> dict[str, list[str]]:
    """
    Function for mapping subjects to their corresponding sessions from input directory.
    """
    mymap = {}
    mypath = os.listdir(os.path.join(mydirectory_name))
    mypath.sort()

    for file in mypath:
        file_path = os.path.join(mydirectory_name, file)
        # iterate through sub-CMH0000{i}
        if os.path.isdir(file_path) and "sub" in file:
            mymap[file] = []
            subj_dir_content = os.listdir(file_path)
            for item in subj_dir_content:
                if "ses" in item:
                    mymap[file].append(item)
            mymap[file].sort()
    return mymap

subject_and_sessions = find_sessions(BASE_DIR)

def extract_NaNs(ts_runs: dict[pd.DataFrame]) -> dict[pd.DataFrame]:
    """Extract NaN ROIs (columns) from a timeseries pd df.
    This function does not mutate and returns a dict where runs are mapped to pd df.
    """
    res = {}
    for run in ts_runs:
        # drop R_PreS_ROI ROI across all subjects since this ROI is sometimes censored and sometimes not in subjects. 
        ts_runs[run].drop("R_PreS_ROI", axis = 1, inplace = True, errors="ignore") # inplace mutates the ts_df value in dict 
        res[run] = ts_runs[run].dropna(axis = 1, how='all') # drop a column containing all NaN values
    return res

def timeseries_df2_corr(myruns : dict[pd.DataFrame]) -> dict[pd.DataFrame]:
    res = {}
    for run_key in myruns:
        df_copy = myruns[run_key].copy()
        res[run_key] = df_copy.corr()
    return res


def vectorize(mycorr_runs: dict[pd.DataFrame]) -> list[list[float]]:
    """
    Return a vector from all the runs present within the subject.
    NOTE: that in the header says list[list[float]] but the function actually returns
    np.ndarray instead of list for better readability of the result. This is left as list[list[float]]
    for readability purposes.
    """
    res = []
    for item in mycorr_runs:
        mycorr_df = mycorr_runs[item]
        n = len(mycorr_runs[item])
        temp = []
        # iterate thru matrix
        for col in mycorr_df:
            for val in mycorr_df[col]:
                if val == float(1):
                    break
                temp.append(val)
        res.append(np.array(temp))
    num_of_unique_vals = n * (n-1) / 2
    # test to ensure that unique vals in the corr_matrix are being captured
    if num_of_unique_vals != len(res[0]):
        raise ValueError # since we are not extracting the unique vals, we can expect something wrong in the map.

    return res


def average_across_runs(vectors_of_runs: list[list[float]]) -> list[float]:
    """ Return mean at index_i across all the vectors present in input.
    Example:
    input = [[1, 2, 3], [2, 4, 5], [9, 8, 10]]
    Function will return array of len==3 s.t.
    array[0] = (1 + 2 + 9) / 3
    array[1] = (2 + 4 + 5) / 3
    array[2] = (3 + 5 + 10) / 3 """
    res = []
    n_runs = len(vectors_of_runs)
    single_run_length = len(vectors_of_runs[0])
    for i in range(single_run_length):
        curr_sum = 0
        for j in range(n_runs):
            curr_sum += vectors_of_runs[j][i]
        res.append(curr_sum/n_runs)
    return pd.array(res)

# k = extract_NaNs(var.ts_of_runs)
# k = timeseries_df2_corr(k)
# k = vectorize(k)


def create_ConnectivityMatrix_across_sessions(subject_name: str) -> dict[str, ConnectivityMatrix]:
    """ Return a dictionary where sessions are mapped to ConnectivityMatrix objects specific to that session within a subject.
    Precondition: subject_name must be existing key in the subject_and_sessions dictionary.
    """
    mymap = {}
    my_sessions = subject_and_sessions[subject_name]
    for ses in my_sessions:
        session_runs_df = ConnectivityMatrix(ses, subject_name).ts_of_runs
        df_nan_extracted = extract_NaNs(session_runs_df)
        roi2roi_corr_matrix = timeseries_df2_corr(df_nan_extracted)
        sub_vectors = vectorize(roi2roi_corr_matrix)
        mymap[ses] = sub_vectors

    return mymap

# create_ConnectivityMatrix_across_sessions("sub-CMH00000010")

def create_subject_to_mean_vectors() -> dict[str, list]:
    res = {}
    for key in subject_and_sessions:
        all_runs = []
        LT_vectors = create_ConnectivityMatrix_across_sessions(key)
        for ses in LT_vectors:
            all_runs.extend(LT_vectors[ses])
        res[key] = average_across_runs(all_runs)
        print(f"{key} has total runs: {len(all_runs)}")
    return res

# final_res = create_subject_to_mean_vectors()

# two functions can be combined into one big logic for efficiency purposes but it is left as seperated for readability. 

def avg_by_runs_in_sessions() -> dict[str, list[list[float]]]:
    """This function does a similar thing as above. However rather than mapping subjects to a 
    mean vector calculated by averaging across every single runs within the subject, it instead
    maps to mean calculated by averaging across every single runs within a session. 
    Example:
    {sub100: [ses_01_runs_avg, ses_02_runs_avg, ses03_runs_avg, ...], 
    sub200: [ses_01_runs_avg, ses_02_runs_avg] 
    ...
    }
    where each ses_0$i_runs_avg is a list[float]. 
    """
    res = {}
    for key in subject_and_sessions:
        res[key] = [] 
        LT_vectors = create_ConnectivityMatrix_across_sessions(key)
        for ses in LT_vectors:
            runs_avg_within_ses = average_across_runs(LT_vectors[ses])
            res[key].append(runs_avg_within_ses)
        print(f"{key} has total sessions: {len(res[key])}")
    return res 

by_sessions_res = avg_by_runs_in_sessions()

# uncomment below if you wanna do the analysis by averaging across all sessions
# by_subjects_res = create_subject_to_mean_vectors() 
# for key in final_res:
#     print(f"{key} has unique vector lenght of {len(final_res[key])}")

def save_as_csv_file(myout):
    mydf = pd.DataFrame.from_dict(myout).tranpose()
    print("enter output name with trailing .csv")
    outname = input() 
    mydf.to_csv(outname)


print("Call save_as_csv_file(by_sessions_res) to save the dictionary output with averaged values across runs as a csv table.")

