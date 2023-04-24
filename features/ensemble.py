import os
import json
import numpy as np

def get_feature_array(json_path, seg_path):
    with open(json_path, "r") as f:
        json_dict = json.load(f)
    
    t1 = []
    for i in json_dict["training"]:
        t1.append("../data/" + i["image"][0])
    id_ls = [i.split("/")[-2].split("_")[-1] for i in t1]

    arr = []
    for id_ in id_ls:
        path = os.path.join(i + ".npy")
        arr.append(np.load(path))
    arr = np.concatenate(arr, axis=0)

    return arr, id_ls