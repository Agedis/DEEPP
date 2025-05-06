
import os


def find_sessions(mydirectory_name: str) -> dict[dict]:
    """
    Function for mapping subjects to their corresponding sessions from input directory. 
    Return a nested dictionary of the form: dict[sub: dict[ses: [func_run_1, func_run_2, ...]]
    Example output: 
    {sub10: {ses01: [run1, run2], ses02: [run1, run2], ...}, 
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
                    session_path = os.path.join(file_path, item, "func")
                    for run in os.listdir(session_path):
                        mymap[file][item].append(run)
                    fmap_path = os.path.join(file_path, item, "fmap")
                    for fmap in os.listdir(fmap_path):
                        if "acq-rest_dir" in fmap:
                            mymap[file][item].append(fmap)
                    mymap[file][item].sort()
    return mymap 


print(f"Please enter the bids path you want to check: ") 

mydir = input()
temp = find_sessions(mydir) 

header = "*" * 60
print(header)
print(f"Printing subject to sessions map")
for key, val in temp.items():
    ses_accumulator = [] 
    for ses in val:
        ses_accumulator.append(ses)
    ses_accumulator.sort()
    print(f"{key} has sessions: {ses_accumulator}")
    

print(header)
print("Printing sub-ses-run map")

print("NOTE: AP + PA is considered as 1 in total fmap runs")

print("|{:^17}| {:^8} | {:^9} | {:^9} |".format("subject_name", "ses_name", "bold_runs", "fmap_runs"))

for key in temp:
    sessions_sorted = list(temp[key])
    sessions_sorted.sort()
    for ses in sessions_sorted:
        func_files = [] 
        fmap_files = []
        for item in temp[key][ses]:
            if "task-rest" in item:
                func_files.append(item)
            elif "acq-rest_dir" in item:
                fmap_files.append(item)
        total_func_files = len(func_files)
        total_fmap_files = len(fmap_files)
        # divide by 2 at the end for func and fmap files since json files are being counted 
        print("|{:^17}| {:^8} | {:^9} | {:^9} |".format(key, ses, total_func_files/2, total_fmap_files/2))
    print(header)





