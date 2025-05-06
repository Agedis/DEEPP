import os
import numpy as np
import pandas as pd
import nibabel as nib



old_path = "/projects/jbyambadorj/func_con_matrix/DEEPPD_old"
new_path = "/projects/jbyambadorj/func_con_matrix/DEEPPD_rerun"

new_deeppi_path = "/projects/jbyambadorj/func_con_matrix/DEEPPI_old"
old_deeppi_path = "/projects/jbyambadorj/func_con_matrix/DEEPPI_rerun"



print("Enter the path you want to check for NaN cols")
print('example path: "/projects/jbyambadorj/func_con_matrix/DEEPPI_rerun"')
my_path = input()

def find_sessions(mydirectory_name: str) -> dict[dict]:
    """
    Function for mapping subjects to their corresponding sessions from input directory.
    Return a nested dictionary of the form: dict[sub: dict[ses: [func_run_1, func_run_2, ...]]
    Example output:
    {sub10: {ses01: [run1, run2], ses02: [run1, run2], ...} ,
    sub09: {ses01: [run1, run2], ses02: [run1, run2], ...},
    ...
    }
    """
    mymap = {}

    mypath = os.listdir(os.path.join(mydirectory_name))
    mypath.sort()

    for file in mypath:
        file_path = os.path.join(mydirectory_name, file)
        ses_to_run_map = {}

        # iterate through sub-CMH0000{i}
        if os.path.isdir(file_path) and "sub" in file:
            mymap[file] = {}
            subj_dir_content = os.listdir(file_path)
            for item in subj_dir_content:
                if "ses" in item:
                    mymap[file][item] = []
                    session_path = os.path.join(file_path, item, "surface")
                    for run in os.listdir(session_path):
                        mymap[file][item].append(run)
                    # uncomment below for checking ROIs in Tian parcellated files
                    # Tian_files = os.path.join(file_path, item, "fmap")
                    # for file in os.listdir(Tian_files):
                    #     mymap[file][item].append(file)
                    mymap[file][item].sort()
    return mymap

temp = find_sessions(my_path)


def create_pd_df(pt_filename: str, basedir: str) -> pd.DataFrame:
    """Return pandas dataframe of the parcellated timeseries file based on the input filename. Directory of the files
    is figured out based on subject identifier in the file name. basedir is either the .../DEEPPD dir or .../DEEPPD_rerun """
    sub_name = pt_filename[:15]
    ses_name = pt_filename[16:22]
    full_path = f"{sub_name}/{ses_name}/surface/{pt_filename}"
    mypath = os.path.join(basedir, full_path)
    foo = nib.load(mypath)
    res = pd.DataFrame(foo.get_fdata(), columns = foo.header.get_axis(1).name)
    return res

def find_na_cols(mydf) -> list[str]:
    """Return a list of na columns (ROIs) in the input pd.DataFrame """
    res = []
    for col in mydf:
        roi_values = mydf[col].to_numpy()
        if True in np.isnan(roi_values):
            res.append(col)
    return res


header = "*" * 50
for key in temp.keys(): 
    print(header)
    
    for ses in sorted(temp[key]):
        for run in sorted(temp[key][ses]):
            mydf = create_pd_df(run, my_path)
            print(f"NA ROIs in the {key}-{ses} are: {find_na_cols(mydf)}")
            # we break here because assumption is that ROIs should be same across runs within session
            break 
