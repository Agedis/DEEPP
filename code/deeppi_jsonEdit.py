from glob import glob
import os
import collections
import json

from bids import BIDSLayout

os.chdir("/mnt/tigrlab/projects/jbyambadorj")


bids_dir = "/projects/jbyambadorj/DEEPPI_fmap_intended"

layout = BIDSLayout(bids_dir, validate=False)

subject_list = layout.get_subjects()
sessions = layout.get_sessions()  # Dynamically get sessions

# Handle session-less data
if not sessions:
    sessions = [None]  
    
# Edit JSON files in fMAP run EPI.jsons

for session in sessions:
    for subject in subject_list:
        # Check if session exists and create session arguments
        session_kwargs = {"session": session} if session else {}
        rest_fmaps = layout.get(subject=subject, acquisition="rest", suffix="epi", extension=".json", return_type="file", **session_kwargs)
        rest_files = layout.get(subject=subject, task="rest", suffix="bold", extension=".nii.gz", return_type = "file", **session_kwargs)
        
        
        intended_files = ['/'.join((i.split('/')[-3:])) for i in rest_files]
        
        fmap_runs = len(rest_fmaps) / 2
        
        if len(rest_fmaps) % 2 == 1:
            print("number of PA and AP types are not matching")
            raise ValueError
        
        if len(rest_files) % 2 == 0:
            my_hash = {} 
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
            my_hash = {}
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
        if len(intended_files) == 0:
            continue 
