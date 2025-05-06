from glob import glob
import os
import collections
import json
from bids import BIDSLayout

os.chdir("/mnt/tigrlab/projects/jbyambadorj")


bids_dir = "/projects/jbyambadorj/DEEP_study/data/raw/DEEPPD_fmap_intended"


# modify the below to specify where the txt file will be generated
mapping_file_path = os.path.join("/projects/jbyambadorj/deep_rerun/bids", "MapDict.txt")

layout = BIDSLayout(bids_dir, validate=False)
subject_list = layout.get_subjects()
sessions = layout.get_sessions()  # Dynamically get sessions
# Handle session-less data
if not sessions:
    sessions = [None]
# Edit JSON files in fMAP run EPI.jsons


f = open("dictionaryMapping.txt", "a")

for session in sessions:
    for subject in subject_list:
        # Check if session exists and create session arguments
        session_kwargs = {"session": session} if session else {}
        rest_fmaps = layout.get(subject=subject, acquisition="rest", suffix="epi", extension=".json", return_type="file", **session_kwargs)
        rest_files = layout.get(subject=subject, task="rest", suffix="bold", extension=".nii.gz", return_type = "file", **session_kwargs)
        intended_files = ['/'.join((i.split('/')[-3:])) for i in rest_files]
        fmap_runs = len(rest_fmaps) / 2


        if len(rest_fmaps) == 0:
            print(f"No fieldmap files for subject {subject} session {session}")
            continue
        

        if len(rest_fmaps) % 2 == 1:
            print(f"Subject {subject}, session {session}: number of PA and AP types are not matching")
            continue

        my_hash = {} 

        if len(rest_files) % 2 == 0:
            # Assume that every fmaps have run number in it
            for i in range(int(fmap_runs)):
                key_name = f"run-0{i + 1}"
                if i == 0:
                    x, y = intended_files[i], intended_files[i+1]
                else:
                    x, y = intended_files[i * 2], intended_files[i*2 + 1]
                my_hash.update({key_name: [x, y]})
        # case where number of func images are odd i.e. missing one last image
        else:
            for i in range(int(fmap_runs) - 1):
                key_name = f"run-0{i + 1}"
                if i == 0:
                    x, y = intended_files[i], intended_files[i+1]
                else:
                    x, y = intended_files[i * 2], intended_files[i*2 + 1]
                my_hash.update({key_name: [x, y]})
            # mapping the last odd image to its corresponding fmap run
            key_name = f"run-0{int(fmap_runs)}"
            my_hash[key_name] = [intended_files[-1]]
        json_identifiers = ["PhaseEncodingDirection", "TotalReadoutTime", "Repeat", "B0FieldSource", "Repeat"]
        f.write(str(my_hash) + "\n")
        

        if len(intended_files) == 0:
            continue
        for rest_fmap in rest_fmaps:
            with open(rest_fmap, 'r+') as j:
                json_dict = json.load(j)
                json_dict_new = json_dict
                PhaseEncodingDirection = json_dict_new["PhaseEncodingDirection"]
                TotalReadoutTime = json_dict_new["TotalReadoutTime"]
                Repeat = json_dict["Repeat"]
                B0FieldSource = json_dict["B0FieldSource"]
                del json_dict_new["PhaseEncodingDirection"]
                del json_dict_new["TotalReadoutTime"]
                del json_dict_new["Repeat"]
                del json_dict_new["B0FieldSource"]
                json_dict_new["Repeat"] = Repeat
                json_dict_new["PhaseEncodingDirection"] = PhaseEncodingDirection
                json_dict_new["TotalReadoutTime"] = TotalReadoutTime
                for key in my_hash.keys():
                    if key in rest_fmap:
                        json_dict_new['IntendedFor'] = my_hash[key]
                        json_dict_new["B0FieldSource"] = B0FieldSource
                        j.seek(0)
                        json.dump(json_dict_new, j, indent=2)
                        j.truncate()

f.close() 
