import os 

print(f"Input your bids directory path: ")
print(f"example: /archive/data/DEEPPI/data/bids")
mydir = input()

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

res = find_sessions(mydir)
for key in res:
    print(f"{key} has sessions: {res[key]}")       

for key in res:
    print(f"{key}: {len(res[key])}") 
